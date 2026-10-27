"""Extract trace data from the x1-370 Neo4j (assistx) database into normalized
training sessions.

The assistx database on x1-370 (Tailscale 100.64.43.123) contains a rich
knowledge graph of fleet operations: TraceGroups, TraceEvents, AgentSessions,
Tasks, ToolCalls, etc. This extractor pulls TraceGroup → TraceEvent chains and
converts them into normalized session records compatible with the auto-finetune
pipeline.

Neo4j connection details:
  - URI: bolt://100.64.43.123:7688 (docker-mapped port)
  - Database: assistx
  - Auth: neo4j / (from NEO4J_PASSWORD env or config)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from src.config import Config
from src.locking import atomic_write_json

# Watermark file for incremental extraction
_WATERMARK_FILE = "neo4j-traces-watermark.json"


def _watermark_path(cfg: Config) -> str:
    """Path to the watermark file tracking last extraction cursor."""
    raw_dir = cfg.path("raw_dir")
    return os.path.join(raw_dir, _WATERMARK_FILE)


def _read_watermark(cfg: Config) -> dict:
    """Read the last extraction watermark."""
    path = _watermark_path(cfg)
    if not os.path.exists(path):
        return {"trace_groups": 0, "tool_calls": 0, "last_ts": None}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"trace_groups": 0, "tool_calls": 0, "last_ts": None}


def _write_watermark(cfg: Config, wm: dict) -> None:
    """Write the extraction watermark."""
    path = _watermark_path(cfg)
    try:
        atomic_write_json(path, wm)
    except Exception:
        pass


# Neo4j connection defaults (overridable via config or env)
DEFAULT_NEO4J_URI = "bolt://100.64.43.123:7688"
DEFAULT_NEO4J_DB = "assistx"
DEFAULT_NEO4J_USER = "neo4j"

# Batch size for Neo4j queries (to avoid memory issues on large graphs)
_BATCH_SIZE = 500
# Maximum trace groups to extract per run
_MAX_TRACE_GROUPS = 50_000


def _get_neo4j_config(cfg: Config) -> dict:
    """Resolve Neo4j connection parameters from config and environment."""
    n4 = cfg.get("sources", "neo4j_traces", default={}) or {}
    uri = n4.get("uri") or os.environ.get("NEO4J_TRACES_URI") or DEFAULT_NEO4J_URI
    db = n4.get("database") or os.environ.get("NEO4J_TRACES_DB") or DEFAULT_NEO4J_DB
    user = n4.get("user") or os.environ.get("NEO4J_TRACES_USER") or DEFAULT_NEO4J_USER
    pw = os.environ.get("NEO4J_TRACES_PASSWORD") or os.environ.get("NEO4J_PASSWORD") or n4.get("password")
    return {"uri": uri, "database": db, "user": user, "password": pw}


def _connect(uri: str, user: str, password: str, db: str = 'neo4j', retries: int = 3):
    """Connect to Neo4j with retry and exponential backoff."""
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        return None, f"neo4j driver unavailable ({e})"

    last_err = None
    for attempt in range(max(1, retries)):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session(database=db) as s:
                s.run("RETURN 1").single()
            return driver, None
        except Exception as e:
            last_err = e
            try:
                driver.close()
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return None, f"neo4j connection failed after {retries} attempts: {last_err}"


def _safe_json(val: Any) -> Any:
    """Safely parse a JSON value."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    if isinstance(val, str):
        return val
    try:
        return str(val)
    except Exception:
        return default


def _safe_neo4j_value(val: Any) -> Any:
    """Convert Neo4j-specific types (DateTime, Date, etc.) to JSON-serializable forms."""
    if val is None:
        return None
    type_name = type(val).__name__
    if type_name in ("DateTime", "Date", "Time", "LocalDateTime",
                     "LocalTime", "Duration", "OffsetTime"):
        try:
            return str(val)
        except Exception:
            return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe_neo4j_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _safe_neo4j_value(v) for k, v in val.items()}
    if hasattr(val, "items"):
        try:
            return {k: _safe_neo4j_value(v) for k, v in val.items()}
        except Exception:
            pass
    try:
        return str(val)
    except Exception:
        return None


def _trace_group_to_session(group: dict, events: list[dict]) -> dict | None:
    """Convert a TraceGroup + its events into a normalized session record."""
    corr_id = group.get("correlation_id", "")
    if not corr_id:
        return None

    messages = []
    for ev in events:
        event_type = ev.get("event_type", "")
        source = ev.get("source", "")
        payload = _safe_json(ev.get("payload_json", ""))
        ts = _safe_neo4j_value(ev.get("ts")) or _safe_neo4j_value(ev.get("created_at")) or ""

        # Build a text representation of the event
        parts = []
        if isinstance(payload, dict):
            # Extract key fields for a readable summary
            summary_fields = []
            for key in ("candidates", "selected_model", "model", "decision",
                        "error", "status", "result", "agent", "task_id"):
                if key in payload and payload[key] is not None:
                    val = payload[key]
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val)[:500]
                    summary_fields.append(f"{key}={val}")
            if summary_fields:
                text = f"[{event_type}] {' | '.join(summary_fields)}"
            else:
                text = f"[{event_type}] {json.dumps(payload)[:500]}"
        else:
            text = f"[{event_type}] {str(payload)[:500]}"

        role = "assistant" if source in ("auto-router", "auto-assign", "auto-assist") else "user"
        messages.append({
            "id": ev.get("event_id", ""),
            "role": role,
            "agent": source,
            "model": "",
            "time": ts,
            "parts": [{"type": "text", "text": text}],
        })

    if len(messages) < 2:
        return None

    return {
        "source": "neo4j-traces",
        "session_id": f"trace_{corr_id}",
        "title": f"Trace {corr_id[:12]}",
        "agent": "neo4j-traces",
        "model": "",
        "project_id": "",
        "directory": "",
        "time_created": _safe_str(events[0].get("ts", 0)) if events else 0,
        "time_updated": _safe_str(events[-1].get("ts", 0)) if events else 0,
        "messages": messages,
    }


def _tool_call_to_session(tc: dict) -> dict | None:
    """Convert a ToolCall record into a normalized training session."""
    tool = tc.get("tool", "")
    if not tool:
        return None
    input_json = _safe_json(tc.get("input_json", ""))
    output_json = _safe_json(tc.get("output_json", ""))
    ok = tc.get("ok", False)
    run_id = tc.get("run_id", "")

    input_text = json.dumps(input_json, indent=2) if isinstance(input_json, (dict, list)) else str(input_json or "")
    output_text = json.dumps(output_json, indent=2) if isinstance(output_json, (dict, list)) else str(output_json or "")

    # Build a conversation: user asks to use tool, assistant uses it, gets result
    messages = [
        {
            "id": f"tc_{tc.get('id', '')}_user",
            "role": "user",
            "agent": "neo4j-traces",
            "model": "",
            "time": tc.get("started_at", ""),
            "parts": [{"type": "text", "text": f"Use the {tool} tool."}],
        },
        {
            "id": f"tc_{tc.get('id', '')}_assistant",
            "role": "assistant",
            "agent": "neo4j-traces",
            "model": "",
            "time": tc.get("started_at", ""),
            "parts": [
                {"type": "tool", "tool": tool, "call_id": tc.get("id", ""),
                 "status": "completed" if ok else "failed",
                 "input": input_json, "output": output_json},
            ],
        },
    ]

    if len(messages) < 2:
        return None

    return {
        "source": "neo4j-traces",
        "session_id": f"toolcall_{tc.get('id', '')}",
        "title": f"Tool: {tool}",
        "agent": "neo4j-traces",
        "model": "",
        "project_id": "",
        "directory": "",
        "time_created": _safe_str(tc.get("started_at", "")),
        "time_updated": _safe_str(tc.get("ended_at", "")),
        "messages": messages,
    }


def extract_tool_calls(cfg: Config, out_dir: str, incremental: bool = False) -> int:
    """Extract ToolCall records from Neo4j into training sessions."""
    neo_cfg = _get_neo4j_config(cfg)
    uri = neo_cfg["uri"]
    db = neo_cfg["database"]
    user = neo_cfg["user"]
    pw = neo_cfg["password"]

    if not pw:
        print("[neo4j-traces] password missing; skipping tool calls")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    driver, err = _connect(uri, user, pw, db, retries=3)
    if driver is None:
        print(f"[neo4j-traces] {err}; skipping tool calls")
        return 0

    written = 0
    skip = 0
    max_calls = cfg.get("sources", "neo4j_traces", "max_tool_calls", default=5000)

    try:
        with driver.session(database=db) as s:
            total = s.run("MATCH (tc:ToolCall) RETURN count(tc) AS c").single()["c"]
            print(f"[neo4j-traces] found {total} tool calls in {db}")

            wm = _read_watermark(cfg) if incremental else {"tool_calls": 0}
            offset = wm.get("tool_calls", 0) if incremental else 0
            inc_cap = 2000 if incremental else max_calls
            cap = min(offset + inc_cap, max_calls, total)
            remaining = cap - offset
            if remaining <= 0:
                print(f"[neo4j-traces] No new tool calls (watermark={offset}, total={total})")
                return 0
            batch_size = 500
            while offset < cap:
                try:
                    calls = list(s.run(
                        "MATCH (tc:ToolCall) "
                        "RETURN tc.id AS id, tc.tool AS tool, tc.input_json AS input_json, "
                        "       tc.output_json AS output_json, tc.ok AS ok, "
                        "       tc.run_id AS run_id, tc.started_at AS started_at, "
                        "       tc.ended_at AS ended_at "
                        "ORDER BY tc.started_at "
                        "SKIP $offset LIMIT $limit",
                        offset=offset, limit=batch_size,
                    ))
                except Exception as e:
                    print(f"[neo4j-traces] tool call query failed at offset {offset}: {e}; stopping")
                    break

                if not calls:
                    break

                for c in calls:
                    tc = {
                        "id": _safe_neo4j_value(c["id"]),
                        "tool": _safe_neo4j_value(c["tool"]),
                        "input_json": _safe_neo4j_value(c["input_json"]),
                        "output_json": _safe_neo4j_value(c["output_json"]),
                        "ok": c["ok"],
                        "run_id": _safe_neo4j_value(c["run_id"]),
                        "started_at": _safe_neo4j_value(c["started_at"]),
                        "ended_at": _safe_neo4j_value(c["ended_at"]),
                    }
                    rec = _tool_call_to_session(tc)
                    if rec is None:
                        skip += 1
                        continue
                    try:
                        atomic_write_json(
                            os.path.join(out_dir, f"neo4j_toolcall_{tc['id']}.json"), rec)
                        written += 1
                    except Exception as e:
                        skip += 1
                        continue

                    if written % 1000 == 0:
                        print(f"[neo4j-traces] {written} tool calls written")

                offset += len(calls)

            print(f"[neo4j-traces] tool calls: {written} written, {skip} skipped")
    finally:
        try:
            driver.close()
        except Exception:
            pass

    return written


def extract_trace_groups(cfg: Config, out_dir: str, incremental: bool = False) -> int:
    """Extract trace groups from Neo4j into normalized session files."""
    neo_cfg = _get_neo4j_config(cfg)
    uri = neo_cfg["uri"]
    db = neo_cfg["database"]
    user = neo_cfg["user"]
    pw = neo_cfg["password"]

    if not pw:
        print("[neo4j-traces] password missing (set NEO4J_TRACES_PASSWORD or NEO4J_PASSWORD); skipping")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    driver, err = _connect(uri, user, pw, db, retries=3)
    if driver is None:
        print(f"[neo4j-traces] {err}; skipping")
        return 0

    written = 0
    skip = 0
    try:
        with driver.session(database=db) as s:
            # Get total count
            total = s.run("MATCH (g:TraceGroup) RETURN count(g) AS c").single()["c"]
            print(f"[neo4j-traces] found {total} trace groups in {db}")

            # Process in batches using SKIP/LIMIT
            wm = _read_watermark(cfg) if incremental else {"trace_groups": 0}
            offset = wm.get("trace_groups", 0) if incremental else 0
            # Cap incremental runs to avoid processing too many records at once
            inc_cap = 5000 if incremental else _MAX_TRACE_GROUPS
            cap = min(offset + inc_cap, _MAX_TRACE_GROUPS, total)
            remaining = cap - offset
            if remaining <= 0:
                print(f"[neo4j-traces] No new trace groups (watermark={offset}, total={total})")
                return 0
            while offset < cap:
                batch_size = min(_BATCH_SIZE, min(total, _MAX_TRACE_GROUPS) - offset)
                try:
                    groups = list(s.run(
                        "MATCH (g:TraceGroup) "
                        "RETURN g.correlation_id AS correlation_id, "
                        "       g.current_state AS current_state, "
                        "       g.created_at AS created_at, "
                        "       g.created_at_ts AS created_at_ts "
                        "ORDER BY g.created_at_ts "
                        "SKIP $offset LIMIT $limit",
                        offset=offset, limit=batch_size,
                    ))
                except Exception as e:
                    print(f"[neo4j-traces] query failed at offset {offset}: {e}; stopping")
                    break

                if not groups:
                    break

                for g_rec in groups:
                    group = {
                        "correlation_id": _safe_neo4j_value(g_rec["correlation_id"]),
                        "current_state": _safe_neo4j_value(g_rec["current_state"]),
                        "created_at": _safe_neo4j_value(g_rec["created_at"]),
                        "created_at_ts": _safe_neo4j_value(g_rec["created_at_ts"]),
                    }
                    corr_id = group["correlation_id"]
                    if not corr_id:
                        skip += 1
                        continue

                    # Fetch events for this group
                    try:
                        events = list(s.run(
                            "MATCH (g:TraceGroup {correlation_id: $cid})-[:HAS_EVENT]->(e:TraceEvent) "
                            "RETURN e.event_id AS event_id, e.event_type AS event_type, "
                            "       e.source AS source, e.payload_json AS payload_json, "
                            "       e.ts AS ts, e.created_at AS created_at "
                            "ORDER BY e.ts, e.created_at",
                            cid=corr_id,
                        ))
                    except Exception as e:
                        print(f"[neo4j-traces] failed to fetch events for {corr_id}: {e}; skipping")
                        skip += 1
                        continue

                    event_list = []
                    for ev in events:
                        event_list.append({
                            "event_id": _safe_neo4j_value(ev["event_id"]),
                            "event_type": _safe_neo4j_value(ev["event_type"]),
                            "source": _safe_neo4j_value(ev["source"]),
                            "payload_json": _safe_neo4j_value(ev["payload_json"]),
                            "ts": _safe_neo4j_value(ev["ts"]),
                            "created_at": _safe_neo4j_value(ev["created_at"]),
                        })

                    rec = _trace_group_to_session(group, event_list)
                    if rec is None:
                        skip += 1
                        continue

                    try:
                        atomic_write_json(
                            os.path.join(out_dir, f"neo4j_trace_{corr_id}.json"), rec)
                        written += 1
                    except Exception as e:
                        print(f"[neo4j-traces] failed to write {corr_id}: {e}; skipping")
                        skip += 1
                        continue

                    if written % 1000 == 0:
                        print(f"[neo4j-traces] {written} sessions written (offset {offset})")

                offset += len(groups)

            print(f"[neo4j-traces] TOTAL: {written} sessions written, {skip} skipped")
    finally:
        try:
            driver.close()
        except Exception:
            pass

    return written


def main(cfg: Config, incremental: bool = False) -> int:
    """Main entry point for Neo4j trace extraction.

    If incremental=True, only extracts records newer than the last watermark.
    """
    neo_cfg = cfg.get("sources", "neo4j_traces", default={}) or {}
    if not neo_cfg.get("enabled", False):
        print("[neo4j-traces] disabled in config; skipping")
        return 0

    raw_dir = cfg.path("raw_dir")
    os.makedirs(raw_dir, exist_ok=True)
    out_dir = os.path.join(raw_dir, "neo4j-traces")
    os.makedirs(out_dir, exist_ok=True)

    wm = _read_watermark(cfg) if incremental else {"trace_groups": 0, "tool_calls": 0, "last_ts": None}
    if incremental:
        print(f"[neo4j-traces] incremental mode: watermark={wm}")

    count = extract_trace_groups(cfg, out_dir, incremental=incremental)
    print(f"[neo4j-traces] trace groups: {count} written")

    tc_count = extract_tool_calls(cfg, out_dir, incremental=incremental)
    print(f"[neo4j-traces] tool calls: {tc_count} written")

    total = count + tc_count
    print(f"[neo4j-traces] TOTAL sessions written: {total}")

    # Update watermark
    _write_watermark(cfg, {
        "trace_groups": count,
        "tool_calls": tc_count,
        "last_ts": time.time(),
        "mode": "incremental" if incremental else "full",
    })
    return total
