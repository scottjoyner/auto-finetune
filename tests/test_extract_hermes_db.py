"""Behavioural tests for Hermes session extraction.

extract_state_db is what produces the Hermes half of the training corpus, and it
had no test that built a real state.db -- only the small pure helpers were
covered. These tests drive it end to end against a real SQLite fixture, because
the bugs that matter here are the ones that only appear once real rows flow
through: a single malformed message used to abort the entire run, and tool
arguments were dropped without any signal.
"""
from __future__ import annotations

import json
import os
import sqlite3

from src import extract_hermes as H


def _state_db(path, sessions, messages):
    """Create a minimal Hermes state.db shaped like the live one.

    Mirrors the real schema where it matters: messages.id is INTEGER PRIMARY
    KEY AUTOINCREMENT (so ordering by id is insertion order) and sessions.id is
    TEXT.
    """
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT,
            title TEXT,
            started_at REAL NOT NULL,
            ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT
        );
    """)
    for s in sessions:
        con.execute(
            "INSERT INTO sessions (id, source, model, title, started_at, ended_at)"
            " VALUES (?,?,?,?,?,?)", s)
    for m in messages:
        con.execute(
            "INSERT INTO messages (id, session_id, role, content, tool_call_id,"
            " tool_calls, tool_name, timestamp, finish_reason, reasoning,"
            " reasoning_content) VALUES (?,?,?,?,?,?,?,?,?,?,?)", m)
    con.commit()
    con.close()


def _msg(mid, sid, role, content=None, *, tool_call_id=None, tool_calls=None,
         tool_name=None, ts=None, reasoning=None, reasoning_content=None):
    return (mid, sid, role, content, tool_call_id,
            json.dumps(tool_calls) if tool_calls is not None else None,
            tool_name, ts if ts is not None else float(mid), None,
            reasoning, reasoning_content)


def _cfg(tmp_path, **extract):
    from src.config import Config
    raw = {"paths": {"raw_dir": str(tmp_path / "raw")},
           "extract": extract}
    os.makedirs(str(tmp_path / "raw"), exist_ok=True)
    return Config(raw)


def _load(out_dir, sid):
    path = os.path.join(str(out_dir), f"hermes_{sid}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


# --- happy path ------------------------------------------------------------

def test_extracts_a_multi_turn_session(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m1", "Title", 100.0, 200.0)],
              [_msg(1, "s1", "user", "hello"),
               _msg(2, "s1", "assistant", "hi there"),
               _msg(3, "s1", "user", "again")])
    n = H.extract_state_db(_cfg(tmp_path), db, str(out))
    assert n == 1
    rec = _load(out, "s1")
    assert rec["source"] == "hermes"
    assert rec["session_id"] == "s1"
    assert rec["agent"] == "cli"          # source is the agent dimension
    assert rec["model"] == "m1"
    assert rec["title"] == "Title"
    assert rec["time_created"] == 100.0
    assert rec["time_updated"] == 200.0
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "user"]
    # Order must follow insertion order.
    assert [m["id"] for m in rec["messages"]] == ["h1", "h2", "h3"]


def test_tool_call_is_paired_with_its_result(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "run ls"),
               _msg(2, "s1", "assistant", None,
                    tool_calls=[{"id": "c1", "function": {"name": "terminal",
                                                          "arguments": '{"cmd":"ls"}'}}]),
               _msg(3, "s1", "tool", "a.txt\nb.txt",
                    tool_call_id="c1", tool_name="terminal")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    rec = _load(out, "s1")
    assistant = rec["messages"][1]
    tool = assistant["parts"][0]
    assert tool["type"] == "tool"
    assert tool["tool"] == "terminal"
    assert tool["input"] == {"cmd": "ls"}
    assert tool["output"] == "a.txt\nb.txt", "tool result not attached"


def test_orphan_tool_result_becomes_standalone_part(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "hi"),
               _msg(2, "s1", "tool", "output-without-a-call",
                    tool_call_id="ghost", tool_name="terminal")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    parts = _load(out, "s1")["messages"][1]["parts"]
    assert parts[0]["type"] == "tool"
    assert parts[0]["output"] == "output-without-a-call"
    assert parts[0]["input"] is None


def test_reasoning_precedes_text_in_parts(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "q"),
               _msg(2, "s1", "assistant", "answer", reasoning="thinking")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    parts = _load(out, "s1")["messages"][1]["parts"]
    assert [p["type"] for p in parts] == ["reasoning", "text"]


def test_reasoning_content_used_when_reasoning_absent(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "q"),
               _msg(2, "s1", "assistant", "a", reasoning_content="from-content")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    parts = _load(out, "s1")["messages"][1]["parts"]
    assert parts[0] == {"type": "reasoning", "text": "from-content"}


# --- robustness ------------------------------------------------------------

def test_malformed_function_string_does_not_abort_the_run(tmp_path):
    """Regression: `"function": "terminal"` (a bare string) made
    _tool_calls_to_parts raise AttributeError, which propagated out of the
    per-message loop and killed the whole extraction -- every session lost, not
    just the malformed one.
    """
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("bad", "cli", "m", None, 1.0, 2.0),
               ("good", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "bad", "user", "q"),
               _msg(2, "bad", "assistant", None,
                    tool_calls=[{"id": "c1", "function": "terminal"}]),
               _msg(3, "bad", "assistant", "recovered"),
               _msg(4, "good", "user", "q"),
               _msg(5, "good", "assistant", "a")])
    n = H.extract_state_db(_cfg(tmp_path), db, str(out))
    assert n == 2, "one malformed message cost us the whole run"
    assert _load(out, "good") is not None


def test_unparseable_arguments_are_not_silently_dropped(tmp_path):
    """A tool call whose arguments will not parse used to become input=None,
    i.e. the corpus trained on a tool call with no arguments at all and nothing
    said so."""
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "q"),
               _msg(2, "s1", "assistant", None,
                    tool_calls=[{"id": "c1", "name": "terminal",
                                 "arguments": "{not valid json"}])])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    tool = _load(out, "s1")["messages"][1]["parts"][0]
    assert tool["input"] is not None, "arguments vanished with no trace"
    assert "not valid json" in str(tool["input"])


def test_duplicate_tool_results_keep_the_first(tmp_path):
    """A retried tool emits two result rows with the same call_id. The pending
    entry was never cleared, so the second silently overwrote the first."""
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "q"),
               _msg(2, "s1", "assistant", None,
                    tool_calls=[{"id": "c1", "name": "terminal",
                                 "arguments": '{"cmd":"ls"}'}]),
               _msg(3, "s1", "tool", "first-result", tool_call_id="c1",
                    tool_name="terminal"),
               _msg(4, "s1", "tool", "second-result", tool_call_id="c1",
                    tool_name="terminal")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    rec = _load(out, "s1")
    outputs = [p["output"] for m in rec["messages"] for p in m["parts"]
               if p["type"] == "tool"]
    assert outputs[0] == "first-result", "duplicate result overwrote the original"


def test_tool_call_before_its_result_is_still_paired(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "s1", "user", "q"),
               _msg(2, "s1", "assistant", None,
                    tool_calls=[{"id": "cA", "name": "t1", "arguments": "{}"},
                                {"id": "cB", "name": "t2", "arguments": "{}"}]),
               _msg(3, "s1", "tool", "out-b", tool_call_id="cB", tool_name="t2"),
               _msg(4, "s1", "tool", "out-a", tool_call_id="cA", tool_name="t1")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    parts = _load(out, "s1")["messages"][1]["parts"]
    got = {p["call_id"]: p["output"] for p in parts if p["type"] == "tool"}
    assert got == {"cA": "out-a", "cB": "out-b"}


# --- filters ---------------------------------------------------------------

def test_min_messages_filter_drops_short_sessions(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("short", "cli", "m", None, 1.0, 2.0),
               ("long", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "short", "user", "only one"),
               _msg(2, "long", "user", "q"),
               _msg(3, "long", "assistant", "a")])
    n = H.extract_state_db(_cfg(tmp_path, min_messages=2), db, str(out))
    assert n == 1
    assert _load(out, "short") is None
    assert _load(out, "long") is not None


def test_exclude_agents_filter(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s-cli", "cli", "m", None, 1.0, 2.0),
               ("s-noise", "noise", "m", None, 1.0, 2.0)],
              [_msg(1, "s-cli", "user", "q"), _msg(2, "s-cli", "assistant", "a"),
               _msg(3, "s-noise", "user", "q"), _msg(4, "s-noise", "assistant", "a")])
    n = H.extract_state_db(_cfg(tmp_path, exclude_agents=["noise"]), db, str(out))
    assert n == 1
    assert _load(out, "s-cli") is not None
    assert _load(out, "s-noise") is None


def test_include_agents_filter(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s-cli", "cli", "m", None, 1.0, 2.0),
               ("s-signal", "signal", "m", None, 1.0, 2.0)],
              [_msg(1, "s-cli", "user", "q"), _msg(2, "s-cli", "assistant", "a"),
               _msg(3, "s-signal", "user", "q"), _msg(4, "s-signal", "assistant", "a")])
    n = H.extract_state_db(_cfg(tmp_path, include_agents=["signal"]), db, str(out))
    assert n == 1
    assert _load(out, "s-signal") is not None
    assert _load(out, "s-cli") is None


def test_sessions_with_no_messages_are_skipped(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("empty", "cli", "m", None, 1.0, 2.0)],
              [_msg(1, "other", "user", "q"), _msg(2, "other", "assistant", "a")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 0


def test_ended_at_falls_back_to_started_at(tmp_path):
    db = str(tmp_path / "state.db")
    out = tmp_path / "raw"
    _state_db(db,
              [("s1", "cli", "m", None, 100.0, None)],
              [_msg(1, "s1", "user", "q"), _msg(2, "s1", "assistant", "a")])
    assert H.extract_state_db(_cfg(tmp_path), db, str(out)) == 1
    assert _load(out, "s1")["time_updated"] == 100.0


# --- helper hardening ------------------------------------------------------

def test_tool_calls_tolerate_non_dict_entries():
    assert H._tool_calls_to_parts(["junk", None, 5]) == []


def test_tool_calls_tolerate_string_function():
    parts = H._tool_calls_to_parts([{"id": "c1", "function": "terminal",
                                     "arguments": '{"a":1}'}])
    assert parts[0]["tool"] == "terminal"


def test_tool_calls_default_call_id_to_none():
    parts = H._tool_calls_to_parts([{"name": "t", "arguments": "{}"}])
    assert parts[0]["call_id"] is None


# --- on-disk directory fallback -------------------------------------------

def test_read_dir_reads_json_and_jsonl(tmp_path):
    src = tmp_path / "sessions"
    src.mkdir()
    (src / "a.json").write_text(json.dumps(
        {"id": "sA", "title": "t", "messages": [{"role": "user", "content": "q"}]}))
    (src / "b.jsonl").write_text(
        json.dumps({"session_id": "sB", "messages": [{"role": "user", "content": "q"}]})
        + "\n")
    (src / "ignored.txt").write_text("nope")
    raw = tmp_path / "raw"
    os.makedirs(str(raw))
    n = H._read_dir({"dir": str(src)}, str(raw))
    assert n == 2
    assert os.path.exists(os.path.join(str(raw), "hermes_sA.json"))
    assert os.path.exists(os.path.join(str(raw), "hermes_sB.json"))


def test_read_dir_skips_corrupt_file_without_aborting(tmp_path):
    src = tmp_path / "sessions"
    src.mkdir()
    (src / "bad.json").write_text("{not json")
    (src / "good.json").write_text(json.dumps(
        {"id": "sG", "messages": [{"role": "user", "content": "q"}]}))
    raw = tmp_path / "raw"
    os.makedirs(str(raw))
    assert H._read_dir({"dir": str(src)}, str(raw)) == 1
    assert os.path.exists(os.path.join(str(raw), "hermes_sG.json"))


def test_read_dir_missing_dir_returns_zero(tmp_path):
    assert H._read_dir({"dir": str(tmp_path / "nope")}, str(tmp_path)) == 0
    assert H._read_dir({}, str(tmp_path)) == 0
