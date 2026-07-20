"""Tests for src.format_dataset."""
from __future__ import annotations

from conftest import make_cfg

from src.format_dataset import (
    _format_window,
    _render_message,
    _render_part,
    _window_messages,
    main,
    to_alpaca,
    to_chatml,
    to_sharegpt,
)


def _msg(role, parts):
    return {"role": role, "parts": parts}


def test_render_text():
    assert _render_part({"type": "text", "text": "hi"}) == "hi"


def test_render_reasoning():
    assert _render_part({"type": "reasoning", "text": "think"}) == "think"


def test_render_tool_with_output():
    p = {"type": "tool", "tool": "bash", "call_id": "c1",
         "input": {"cmd": "ls"}, "output": "file"}
    out = _render_part(p)
    assert "<tool_call" in out and "bash" in out
    assert "<tool_result>" in out and "file" in out


def test_render_tool_no_output():
    p = {"type": "tool", "tool": "x", "call_id": "c", "input": "i"}
    out = _render_part(p)
    assert "<tool_result>" not in out


def test_render_patch():
    out = _render_part({"type": "patch", "files": ["a.py", "b.py"]})
    assert "a.py" in out and "b.py" in out


def test_render_unknown_is_empty():
    assert _render_part({"type": "step-start"}) == ""


def test_render_message_joins():
    m = _msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert _render_message(m) == "a\nb"


def test_to_chatml():
    msgs = [_msg("user", [{"type": "text", "text": "q"}]),
            _msg("assistant", [{"type": "text", "text": "a"}])]
    out = to_chatml(msgs, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["role"] == "user" and out[2]["role"] == "assistant"


def test_to_chatml_skips_empty():
    msgs = [_msg("assistant", [{"type": "step-start"}])]
    out = to_chatml(msgs, "")
    # empty system is preserved; only the content-less assistant is dropped
    assert out == [{"role": "system", "content": ""}]


def test_to_sharegpt():
    msgs = [_msg("user", [{"type": "text", "text": "q"}]),
            _msg("assistant", [{"type": "text", "text": "a"}])]
    out = to_sharegpt(msgs, "SYS")
    assert out[0]["from"] == "system"
    assert out[1]["from"] == "human" and out[2]["from"] == "gpt"


def test_to_alpaca():
    msgs = [_msg("user", [{"type": "text", "text": "q1"}]),
            _msg("assistant", [{"type": "text", "text": "a1"}]),
            _msg("user", [{"type": "text", "text": "q2"}]),
            _msg("assistant", [{"type": "text", "text": "a2"}])]
    out = to_alpaca(msgs, "SYS")
    assert out["instruction"].startswith("SYS")
    assert out["input"] == "q2"
    assert out["output"] == "a2"


def test_window_by_turns():
    msgs = [_msg("user", [{"type": "text", "text": str(i)}]) for i in range(10)]
    wins = _window_messages(msgs, max_turns=4, max_chars=0)
    # step = max_turns//2 = 2 -> windows start at 0,2,4,6,8 => 5 windows
    assert len(wins) == 5
    assert all(len(w) <= 4 for w in wins)


def test_window_by_chars():
    # each message ~100 chars; budget 250 -> ~2-3 per window
    msgs = [_msg("user", [{"type": "text", "text": "x" * 100}]) for _ in range(10)]
    wins = _window_messages(msgs, max_turns=0, max_chars=250)
    assert len(wins) > 1
    for w in wins:
        total = sum(len(_render_part(p)) for m in w for p in m["parts"])
        assert total <= 250


def test_window_whole_when_no_budget():
    msgs = [_msg("user", [{"type": "text", "text": "x"}]) for _ in range(5)]
    wins = _window_messages(msgs, max_turns=0, max_chars=0)
    assert len(wins) == 1 and len(wins[0]) == 5


def test_format_window_dispatch():
    msgs = [_msg("user", [{"type": "text", "text": "q"}])]
    assert "messages" in _format_window(msgs, "chatml", "s")
    assert "conversations" in _format_window(msgs, "sharegpt", "s")
    assert "instruction" in _format_window(msgs, "alpaca", "s")


def test_main_writes_jsonl(tmp_root, sample_session):
    import json
    cleaned = tmp_root / "data" / "cleaned"
    datasets = tmp_root / "data" / "datasets"
    (cleaned / "a.json").write_text(json.dumps(sample_session))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(datasets)})
    n = main(cfg)
    assert n >= 1
    lines = (datasets / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == n
    obj = json.loads(lines[0])
    assert "messages" in obj
