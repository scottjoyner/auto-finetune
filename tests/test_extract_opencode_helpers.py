"""Coverage tests for pure-logic helpers in extract_opencode."""
from __future__ import annotations

from src import extract_opencode as E


def test_safe_json_variants():
    assert E._safe_json('{"a": 1}') == {"a": 1}
    assert E._safe_json("not json") is None
    assert E._safe_json(None) is None
    assert E._safe_json({"already": "dict"}) is None  # str-only variant


def test_build_part_text():
    assert E._build_part({"type": "text", "text": "hi"}) == {"type": "text", "text": "hi"}


def test_build_part_reasoning():
    assert E._build_part({"type": "reasoning", "text": "think"}) == {
        "type": "reasoning", "text": "think"}


def test_build_part_tool():
    d = {"type": "tool", "tool": "bash", "callID": "c1",
         "state": {"status": "completed", "input": {"cmd": "ls"}, "output": "out"}}
    assert E._build_part(d) == {
        "type": "tool", "tool": "bash", "call_id": "c1",
        "status": "completed", "input": {"cmd": "ls"}, "output": "out"}


def test_build_part_patch():
    d = {"type": "patch", "hash": "h1", "files": ["a.py"]}
    assert E._build_part(d) == {"type": "patch", "hash": "h1", "files": ["a.py"]}


def test_build_part_other_marker():
    d = {"type": "step-start", "ts": 1}
    assert E._build_part(d) == {"type": "step-start", "ts": 1}
