"""Extract opencode sessions into normalized JSON conversation files.

Output per session (data/raw/<session_id>.json):
{
  "source": "opencode",
  "session_id": "...",
  "title": "...",
  "agent": "...",
  "model": "...",
  "project": "...",
  "time_created": 123,
  "messages": [
     {
       "id": "...",
       "role": "user"|"assistant",
       "agent": "...",
       "model": "...",
       "time": 123,
       "parts": [ {"type": "text", "text": "..."},
                  {"type": "tool", "tool": "...", "input": {...}, "output": "..."},
                  {"type": "patch", ...},
                  {"type": "reasoning", "text": "..."} ]
     }, ...
  ]
}
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.db import CorruptDB
from src.config import Config


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _build_part(d: dict) -> dict:
    ptype = d.get("type")
    part: dict[str, Any] = {"type": ptype}
    if ptype == "text":
        part["text"] = d.get("text", "")
    elif ptype == "reasoning":
        part["text"] = d.get("text", "")
    elif ptype == "tool":
        st = d.get("state", {}) or {}
        part["tool"] = d.get("tool")
        part["call_id"] = d.get("callID")
        part["status"] = st.get("status")
        part["input"] = st.get("input")
        part["output"] = st.get("output")
    elif ptype == "patch":
        part["hash"] = d.get("hash")
        part["files"] = d.get("files")
    else:
        # step-start / step-finish / compaction -> lightweight markers
        part.update({k: v for k, v in d.items() if k != "type"})
    return part


def extract_db(cfg: Config, db_path: str, out_dir: str, progress_every: int = 10) -> int:
    print(f"[extract] opening {db_path}")
    db = CorruptDB(db_path)

    # Sessions (small table; read by rowid to survive corruption).
    session_cols = ("id", "project_id", "workspace_id", "parent_id", "slug", "directory", "path",
                    "title", "version", "share_url", "metadata", "cost", "tokens_input",
                    "tokens_output", "tokens_reasoning", "tokens_cache_read", "tokens_cache_write",
                    "revert", "permission", "agent", "model", "time_created", "time_updated",
                    "time_compacting", "time_archived")
    sessions = {r[0]: r for r in db.iter_rows("session", session_cols)}
    print(f"[extract] found {len(sessions)} sessions in {os.path.basename(db_path)}")

    excl = set(cfg.get("extract", "exclude_agents", default=[]) or [])
    incl = set(cfg.get("extract", "include_agents", default=[]) or [])
    min_msgs = cfg.get("extract", "min_messages", default=2)

    # One pass over messages + parts, bucketed by session.
    msgs: dict[str, dict] = {}          # message_id -> record
    msg_order: dict[str, list] = {}     # session_id -> [message_id,...]
    msg_by_session: dict[str, dict] = {}  # session_id -> {mid: record}

    total_msgs = 0
    for mid, s_id, t_created, t_updated, data in db.iter_rows(
        "message", ("id", "session_id", "time_created", "time_updated", "data")
    ):
        total_msgs += 1
        if s_id not in sessions:
            continue
        d = _safe_json(data)
        if d is None:
            continue
        m = {
            "id": mid,
            "role": d.get("role"),
            "agent": d.get("agent"),
            "model": d.get("model", {}).get("modelID") if isinstance(d.get("model"), dict) else d.get("model"),
            "provider": d.get("model", {}).get("providerID") if isinstance(d.get("model"), dict) else None,
            "time": (d.get("time") or {}).get("created", t_created),
            "parent_id": d.get("parentID"),
            "parts": [],
        }
        msgs[mid] = m
        msg_order.setdefault(s_id, []).append(mid)
        msg_by_session.setdefault(s_id, {})[mid] = m

    total_parts = 0
    for pid, mid, s_id, t_created, t_updated, data in db.iter_rows(
        "part", ("id", "message_id", "session_id", "time_created", "time_updated", "data")
    ):
        total_parts += 1
        if mid not in msgs:
            continue
        d = _safe_json(data)
        if d is None:
            continue
        msgs[mid]["parts"].append(_build_part(d))

    print(f"[extract] scanned {total_msgs} messages, {total_parts} parts")

    written = 0
    for sid, srow in sessions.items():
        if sid not in msg_order:
            continue
        agent = srow[19] or ""
        if excl and agent in excl:
            continue
        if incl and agent not in incl:
            continue
        ordered = [msg_by_session[sid][m] for m in msg_order[sid]]
        if len(ordered) < min_msgs:
            continue
        rec = {
            "source": "opencode",
            "session_id": sid,
            "title": srow[7],
            "agent": agent,
            "model": srow[20],
            "project_id": srow[1],
            "directory": srow[5],
            "time_created": srow[21],
            "time_updated": srow[22],
            "messages": ordered,
        }
        out_path = os.path.join(out_dir, f"{sid}.json")
        with open(out_path, "w") as f:
            json.dump(rec, f)
        written += 1
        if written % progress_every == 0:
            print(f"[extract] {written} sessions written")
    db.close()
    print(f"[extract] wrote {written} sessions from {os.path.basename(db_path)}")
    return written


def main(cfg: Config) -> int:
    raw_dir = cfg.path("raw_dir")
    os.makedirs(raw_dir, exist_ok=True)
    total = 0
    oc = cfg.get("sources", "opencode", default={})
    main_db = oc.get("db_path")
    if main_db and os.path.exists(main_db):
        total += extract_db(cfg, main_db, raw_dir)
    for extra in (oc.get("extra_dbs") or []):
        if not extra.get("enabled", False):
            continue
        p = extra.get("path")
        if p and os.path.exists(p):
            total += extract_db(cfg, p, raw_dir)
    print(f"[extract] TOTAL sessions written: {total}")
    return total
