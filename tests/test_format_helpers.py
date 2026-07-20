"""Coverage tests for pure-logic helpers in format_dataset + extract_opencode.

These exercise the non-model, non-IO code paths (format conversion, tool-call
extraction, part building) so the data pipeline is verified without a GPU or a
live database.
"""
from __future__ import annotations

import json

from src import extract_opencode as E
from src import format_dataset as F


# ── format_dataset.to_hermes ──────────────────────────────────────────────────
def test_to_hermes_basic_user():
    msgs = [{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
    out = F.to_hermes(msgs, system="SYS")
    assert out["messages"][0] == {"role": "system", "content": "SYS"}
    assert out["messages"][1] == {"role": "user", "content": "hello"}


def test_to_hermes_default_system():
    out = F.to_hermes([], system="")
    assert out["messages"][0] == {"role": "system",
                                  "content": "You are a helpful assistant."}


def test_to_hermes_assistant_tool_call():
    msgs = [{
        "role": "assistant",
        "parts": [{"type": "tool", "tool": "bash", "call_id": "c1",
                   "input": {"command": "ls"}, "output": "file"}],
    }]
    out = F.to_hermes(msgs, system="")
    a = out["messages"][1]
    assert a["role"] == "assistant"
    assert "tool_calls" in a
    assert a["tool_calls"][0]["function"]["name"] == "bash"


def test_to_hermes_tool_result_role():
    msgs = [{"role": "tool", "parts": [{"type": "tool", "tool": "bash",
                                        "call_id": "c1", "output": "OUT"}]}]
    out = F.to_hermes(msgs, system="")
    tr = out["messages"][1]
    assert tr["role"] == "tool"
    assert tr["content"] == "OUT"
    assert tr["tool_call_id"] == "c1"


def test_to_hermes_skips_empty_user():
    msgs = [{"role": "user", "parts": [{"type": "text", "text": ""}]}]
    out = F.to_hermes(msgs, system="S")
    # only the system message, the empty user turn is dropped
    assert len(out["messages"]) == 1


# ── format_dataset._extract_tool_* ─────────────────────────────────────────────
def test_extract_tool_calls():
    m = {"role": "assistant", "parts": [
        {"type": "tool", "tool": "write", "call_id": "x", "input": {"path": "a"}}]}
    calls = F._extract_tool_calls(m)
    assert calls[0]["function"]["name"] == "write"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a"}


def test_extract_tool_results():
    m = {"role": "tool", "parts": [
        {"type": "tool", "tool": "bash", "call_id": "c1", "output": "OUT"}]}
    res = F._extract_tool_results(m)
    assert res[0]["role"] == "tool"
    assert res[0]["content"] == "OUT"


def test_extract_tool_results_stringifies_non_str():
    m = {"role": "tool", "parts": [
        {"type": "tool", "tool": "t", "call_id": "c", "output": {"k": 1}}]}
    res = F._extract_tool_results(m)
    assert res[0]["content"] == '{"k": 1}'


# ── extract_opencode helpers ───────────────────────────────────────────────────
def test_safe_json_roundtrip():
    assert E._safe_json('{"a": 1}') == {"a": 1}
    assert E._safe_json("not json") is None


def test_build_part_text_and_reasoning():
    assert E._build_part({"type": "text", "text": "hi"}) == {"type": "text", "text": "hi"}
    assert E._build_part({"type": "reasoning", "text": "thinking"}) == {
        "type": "reasoning", "text": "thinking"}


def test_build_part_tool():
    d = {"type": "tool", "tool": "bash", "callID": "c9",
         "state": {"status": "ok", "input": {"command": "ls"}, "output": "out"}}
    p = E._build_part(d)
    assert p["tool"] == "bash"
    assert p["call_id"] == "c9"
    assert p["status"] == "ok"
    assert p["input"] == {"command": "ls"}
    assert p["output"] == "out"


def test_build_part_patch():
    p = E._build_part({"type": "patch", "hash": "h1", "files": ["a.py"]})
    assert p == {"type": "patch", "hash": "h1", "files": ["a.py"]}


def test_build_part_marker_passthrough():
    # step-start / compaction markers keep their other keys
    p = E._build_part({"type": "step-start", "step": 3, "token": 7})
    assert p["step"] == 3 and p["token"] == 7
