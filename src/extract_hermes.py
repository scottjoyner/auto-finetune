"""Harvest Hermes agent sessions into the same normalized JSON format as
``extract_opencode``.

Canonical source is the Hermes SQLite session store (``state.db``) — the system
of record with the complete session/message history. Optionally enriched from
the live Neo4j knowledge graph for sessions/nodes not present in state.db.

state.db schema (relevant):
  sessions(id, source, model, system_prompt, parent_session_id, started_at,
           ended_at, end_reason, title, message_count, tool_call_count, ...)
  messages(id, session_id, role, content, tool_call_id, tool_calls, tool_name,
           timestamp, finish_reason, reasoning, reasoning_content, ...)

Output per session matches extract_opencode: a normalized record with
``messages: [{id, role, agent, model, time, parts:[...]}]`` where parts are
text / reasoning / tool blocks.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from src.config import Config


def _adapt_record(rec: dict) -> dict | None:
    """Normalize a raw Hermes session dict into the shared record schema.

    Used by the on-disk JSON/JSONL reader. Returns ``None`` when there are no
    messages to train on (e.g. empty conversation).
    """
    msgs = rec.get("messages")
    if not msgs:
        return None
    sid = rec.get("id") or rec.get("session_id") or ""
    return {
        "source": "hermes",
        "session_id": sid,
        "title": rec.get("title") or "",
        "agent": rec.get("agent") or rec.get("source") or "",
        "model": rec.get("model") or "",
        "project_id": rec.get("project_id") or "",
        "directory": rec.get("directory") or "",
        "time_created": rec.get("time_created") or 0,
        "time_updated": rec.get("time_updated") or 0,
        "messages": msgs,
    }


def _read_dir(h: dict, raw_dir: str) -> int:
    """Read Hermes sessions exported as JSON/JSONL files in a directory.

    Fallback/testing path used when no ``state_db`` is configured.
    """
    d = h.get("dir")
    if not d or not os.path.isdir(d):
        return 0
    written = 0
    for fn in sorted(os.listdir(d)):
        if not (fn.endswith(".json") or fn.endswith(".jsonl")):
            continue
        path = os.path.join(d, fn)
        try:
            if fn.endswith(".jsonl"):
                with open(path) as f:
                    rows = [json.loads(l) for l in f if l.strip()]
            else:
                with open(path) as f:
                    rows = [json.load(f)]
        except Exception:
            continue
        for rec in rows:
            adapted = _adapt_record(rec)
            if adapted is None:
                continue
            sid = adapted["session_id"] or fn
            with open(os.path.join(raw_dir, f"hermes_{sid}.json"), "w") as f:
                json.dump(adapted, f)
            written += 1
    print(f"[hermes] read {written} sessions from dir {d}")
    return written


def _safe_json(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except Exception:
        return None


def _tool_calls_to_parts(tool_calls_raw: Any) -> list[dict]:
    """Normalize an assistant message's tool_calls JSON into tool parts."""
    tc = _safe_json(tool_calls_raw)
    parts: list[dict] = []
    if not tc:
        return parts
    if isinstance(tc, dict):
        tc = [tc]
    for call in tc:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = call.get("name") or fn.get("name")
        args = call.get("arguments")
        if args is None:
            args = fn.get("arguments")
        args = _safe_json(args) if isinstance(args, str) else args
        parts.append({
            "type": "tool",
            "tool": name,
            "call_id": call.get("id") or call.get("tool_call_id"),
            "status": "completed",
            "input": args,
            "output": None,  # filled in from the paired tool-result message
        })
    return parts


def extract_state_db(cfg: Config, db_path: str, out_dir: str) -> int:
    print(f"[hermes] opening {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    excl = set(cfg.get("extract", "exclude_agents", default=[]) or [])
    incl = set(cfg.get("extract", "include_agents", default=[]) or [])
    min_msgs = cfg.get("extract", "min_messages", default=2)

    sessions = {r["id"]: r for r in cur.execute("SELECT * FROM sessions")}
    print(f"[hermes] found {len(sessions)} sessions")

    # Bucket messages by session, ordered by id (insertion order).
    by_session: dict[str, list[sqlite3.Row]] = {}
    for m in cur.execute(
        "SELECT id, session_id, role, content, tool_call_id, tool_calls, "
        "tool_name, timestamp, finish_reason, reasoning, reasoning_content "
        "FROM messages ORDER BY session_id, id"
    ):
        by_session.setdefault(m["session_id"], []).append(m)

    written = 0
    for sid, srow in sessions.items():
        rows = by_session.get(sid)
        if not rows:
            continue
        source = (srow["source"] or "")
        # `source` here is cron/cli/signal, used as the "agent" dimension.
        if excl and source in excl:
            continue
        if incl and source not in incl:
            continue

        messages: list[dict] = []
        # Map tool_call_id -> the tool part awaiting its output.
        pending_tool: dict[str, dict] = {}

        for m in rows:
            role = m["role"]
            if role == "tool":
                # Tool result: attach output to the matching assistant tool part.
                out_val = m["content"]
                cid = m["tool_call_id"]
                target = pending_tool.get(cid) if cid else None
                if target is not None:
                    target["output"] = out_val
                else:
                    # Orphan tool result: emit a standalone tool part.
                    messages.append({
                        "id": f"h{m['id']}",
                        "role": "assistant",
                        "agent": source,
                        "model": srow["model"],
                        "time": m["timestamp"],
                        "parts": [{
                            "type": "tool",
                            "tool": m["tool_name"],
                            "call_id": cid,
                            "status": "completed",
                            "input": None,
                            "output": out_val,
                        }],
                    })
                continue

            parts: list[dict] = []
            reasoning = m["reasoning"] or m["reasoning_content"]
            if reasoning:
                parts.append({"type": "reasoning", "text": reasoning})
            if m["content"]:
                parts.append({"type": "text", "text": m["content"]})
            tool_parts = _tool_calls_to_parts(m["tool_calls"])
            for tp in tool_parts:
                parts.append(tp)
                if tp.get("call_id"):
                    pending_tool[tp["call_id"]] = tp

            if not parts:
                continue
            messages.append({
                "id": f"h{m['id']}",
                "role": role,
                "agent": source,
                "model": srow["model"],
                "time": m["timestamp"],
                "parts": parts,
            })

        if len(messages) < min_msgs:
            continue

        rec = {
            "source": "hermes",
            "session_id": sid,
            "title": srow["title"] or "",
            "agent": source,
            "model": srow["model"],
            "project_id": "",
            "directory": "",
            "time_created": srow["started_at"] or 0,
            "time_updated": srow["ended_at"] or srow["started_at"] or 0,
            "messages": messages,
        }
        with open(os.path.join(out_dir, f"hermes_{sid}.json"), "w") as f:
            json.dump(rec, f)
        written += 1
        if written % 200 == 0:
            print(f"[hermes] {written} sessions written")

    con.close()
    print(f"[hermes] wrote {written} sessions from state.db")
    return written


def enrich_from_neo4j(cfg: Config, out_dir: str) -> int:
    """Optionally pull sessions that live in Neo4j but not in state.db.

    Disabled by default. Requires the `neo4j` driver and a reachable graph.
    Only imports KgSession/KgNode subtrees whose session_id has no state.db
    export already on disk (state.db is canonical for overlaps).
    """
    n4 = cfg.get("sources", "hermes", "neo4j", default={}) or {}
    if not n4.get("enabled", False):
        print("[hermes] neo4j enrichment disabled; skipping")
        return 0
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        print(f"[hermes] neo4j driver unavailable ({e}); skipping enrichment")
        return 0

    uri = n4.get("uri")
    user = n4.get("user", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD") or n4.get("password")
    if not (uri and pw):
        print("[hermes] neo4j uri/password missing (set NEO4J_PASSWORD); skipping")
        return 0

    existing = {fn[len("hermes_"):-len(".json")]
                for fn in os.listdir(out_dir)
                if fn.startswith("hermes_") and fn.endswith(".json")}

    written = 0
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        with driver.session() as s:
            sess_ids = [r["sid"] for r in s.run(
                "MATCH (x:KgSession) RETURN x.session_id AS sid"
            ) if r["sid"]]
            for sid in sess_ids:
                if sid in existing:
                    continue  # state.db is canonical
                rows = list(s.run(
                    "MATCH (x:KgSession {session_id:$sid})-[:HAS]->(m:KgNode) "
                    "RETURN m.kind AS kind, m.role AS role, m.text AS text, "
                    "m.tool AS tool, m.input AS input, m.output AS output, "
                    "m.created_at AS t ORDER BY m.seq, m.created_at",
                    sid=sid,
                ))
                messages: list[dict] = []
                for r in rows:
                    kind = r["kind"]
                    if kind in ("message",):
                        parts = [{"type": "text", "text": r["text"] or ""}]
                        role = r["role"] or "assistant"
                    elif kind == "reasoning":
                        parts = [{"type": "reasoning", "text": r["text"] or ""}]
                        role = "assistant"
                    elif kind in ("toolcall", "toolresult"):
                        parts = [{"type": "tool", "tool": r["tool"],
                                  "input": _safe_json(r["input"]),
                                  "output": r["output"]}]
                        role = "assistant"
                    else:
                        continue
                    messages.append({"id": None, "role": role, "agent": "",
                                     "model": "", "time": r["t"], "parts": parts})
                if len(messages) < 2:
                    continue
                rec = {"source": "hermes", "session_id": sid, "title": "",
                       "agent": "neo4j", "model": "", "project_id": "",
                       "directory": "", "time_created": 0, "time_updated": 0,
                       "messages": messages}
                with open(os.path.join(out_dir, f"hermes_{sid}.json"), "w") as f:
                    json.dump(rec, f)
                written += 1
    finally:
        driver.close()
    print(f"[hermes] neo4j enrichment wrote {written} sessions not in state.db")
    return written


def main(cfg: Config) -> int:
    h = cfg.get("sources", "hermes", default={}) or {}
    if not h.get("enabled", False):
        print("[hermes] disabled in config; skipping")
        return 0
    raw_dir = cfg.path("raw_dir")
    os.makedirs(raw_dir, exist_ok=True)

    total = 0
    state_db = h.get("state_db")
    if state_db and os.path.exists(state_db):
        total += extract_state_db(cfg, state_db, raw_dir)
    elif h.get("dir"):
        total += _read_dir(h, raw_dir)
    else:
        print(f"[hermes] state_db not found: {state_db}")

    total += enrich_from_neo4j(cfg, raw_dir)
    print(f"[hermes] TOTAL sessions written: {total}")
    return total
