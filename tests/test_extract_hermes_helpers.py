"""Coverage tests for pure-logic helpers in extract_hermes."""
from __future__ import annotations

from src import extract_hermes as H


def test_adapt_record_normalizes():
    rec = {"id": "s1", "title": "T", "model": "m", "messages": [{"role": "user"}]}
    out = H._adapt_record(rec)
    assert out["source"] == "hermes"
    assert out["session_id"] == "s1"
    assert out["messages"] == [{"role": "user"}]


def test_adapt_record_uses_session_id_fallback():
    rec = {"session_id": "sess-9", "messages": [{"role": "assistant"}]}
    out = H._adapt_record(rec)
    assert out["session_id"] == "sess-9"


def test_adapt_record_none_when_no_messages():
    assert H._adapt_record({"id": "x"}) is None
    assert H._adapt_record({"messages": []}) is None


def test_safe_json_variants():
    assert H._safe_json(None) is None
    assert H._safe_json({"a": 1}) == {"a": 1}
    assert H._safe_json('{"b": 2}') == {"b": 2}
    assert H._safe_json("not json") is None


def test_tool_calls_to_parts_from_list():
    raw = [{"id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
    parts = H._tool_calls_to_parts(raw)
    assert parts[0]["type"] == "tool"
    assert parts[0]["tool"] == "bash"
    assert parts[0]["call_id"] == "c1"
    assert parts[0]["input"] == {"command": "ls"}
    assert parts[0]["output"] is None


def test_tool_calls_to_parts_from_dict_and_name_field():
    raw = {"name": "write", "arguments": {"path": "a"}, "id": "c2"}
    parts = H._tool_calls_to_parts(raw)
    assert parts[0]["tool"] == "write"
    assert parts[0]["input"] == {"path": "a"}


def test_tool_calls_to_parts_empty():
    assert H._tool_calls_to_parts(None) == []
    assert H._tool_calls_to_parts("bad json") == []
    assert H._tool_calls_to_parts([]) == []
