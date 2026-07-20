"""Coverage tests for pure-logic dataset-formatting helpers."""
from __future__ import annotations

from src import format_dataset as F


def _msg(role, parts):
    return {"role": role, "parts": parts}


def test_render_part_text():
    assert F._render_part({"type": "text", "text": "hi"}) == "hi"


def test_render_part_tool_with_output():
    p = {"type": "tool", "tool": "read", "call_id": "c1",
         "input": {"path": "a"}, "output": "contents"}
    rendered = F._render_part(p)
    assert '<tool_call name="read" call_id="c1">' in rendered
    assert "contents" in rendered


def test_render_part_tool_without_output():
    p = {"type": "tool", "tool": "write", "input": {"x": 1}}
    rendered = F._render_part(p)
    assert "<tool_result>" not in rendered


def test_render_part_patch():
    assert "a.py" in F._render_part({"type": "patch", "files": ["a.py"]})


def test_render_message_joins_parts():
    m = _msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert F._render_message(m) == "a\nb"


def test_to_chatml():
    msgs = [_msg("user", [{"type": "text", "text": "hi"}]),
            _msg("assistant", [{"type": "text", "text": "yo"}])]
    out = F.to_chatml(msgs, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert {"role": "user", "content": "hi"} in out
    assert {"role": "assistant", "content": "yo"} in out


def test_to_sharegpt():
    msgs = [_msg("user", [{"type": "text", "text": "hi"}]),
            _msg("assistant", [{"type": "text", "text": "yo"}])]
    out = F.to_sharegpt(msgs, "SYS")
    assert out[0] == {"from": "system", "value": "SYS"}
    assert {"from": "human", "value": "hi"} in out
    assert {"from": "gpt", "value": "yo"} in out


def test_to_alpaca():
    msgs = [_msg("user", [{"type": "text", "text": "q"}]),
            _msg("assistant", [{"type": "text", "text": "a"}])]
    out = F.to_alpaca(msgs, "SYS")
    assert out["instruction"].endswith("q")
    assert out["output"] == "a"


def test_window_messages_none():
    msgs = [{"parts": []} for _ in range(3)]
    assert F._window_messages(msgs, 0, 0) == [msgs]


def test_window_messages_max_turns():
    msgs = [{"parts": []} for _ in range(5)]
    wins = F._window_messages(msgs, 2, 0)
    assert all(len(w) <= 2 for w in wins)
    assert wins[0] == msgs[:2]


def test_window_messages_max_chars():
    msgs = [{"parts": [{"type": "text", "text": "x" * 10}]} for _ in range(3)]
    wins = F._window_messages(msgs, 0, 15)
    assert len(wins) >= 1


def test_format_window_all_templates():
    msgs = [_msg("user", [{"type": "text", "text": "hi"}])]
    assert "messages" in F._format_window(msgs, "chatml", "S")
    assert "instruction" in F._format_window(msgs, "alpaca", "S")
    assert "conversations" in F._format_window(msgs, "sharegpt", "S")
    assert "messages" in F._format_window(msgs, "hermes", "S")
