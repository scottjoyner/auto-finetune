"""Tests for src.format_dataset."""
from __future__ import annotations

import json

from conftest import make_cfg

from src.format_dataset import (
    _format_window,
    _render_message,
    _render_part,
    _window_messages,
    emit_strata,
    main,
    to_alpaca,
    to_chatml,
    to_sharegpt,
)


def _msg(role, parts):
    return {"role": role, "parts": parts}


def _tool(name, inp=None, out=None):
    p = {"type": "tool", "tool": name}
    if inp is not None:
        p["input"] = inp
    if out is not None:
        p["output"] = out
    return p


def _text(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


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
    # main writes several per-source files plus a merged train.jsonl; assert the
    # merged file is non-empty and parses (not that its line count equals the
    # aggregate `total`, which counts every per-source output).
    assert len(lines) >= 1
    obj = json.loads(lines[0])
    assert "messages" in obj


def _write_session(path, sid, bucket, messages):
    path.write_text(json.dumps(
        {"source": "opencode", "session_id": sid, "messages": messages}))
    return {sid: {"bucket": bucket}}


def test_emit_strata_writes_per_bucket(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    bm.update(_write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run the tests"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}]))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create an add module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/add.py", "content": "def add(a,b):\n    return a+b"}, "ok")]},
         _text("assistant", "Done.")]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out))
    assert counts["shell"] >= 1 and counts["file-edit"] >= 1
    assert (out / "train.shell.jsonl").exists()
    assert (out / "train.file-edit.jsonl").exists()
    lines = (out / "train.shell.jsonl").read_text().strip().splitlines()
    assert len(lines) == counts["shell"]
    import json as _json
    _json.loads(lines[0])


def test_emit_strata_balance_upsamples(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    bm.update(_write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run it"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "ls"}, "ok")]},
         _text("assistant", "done")]))
    for i in range(2):
        bm.update(_write_session(
            cleaned / f"e{i}.json", f"e{i}", "file-edit",
            [_text("user", "create module"),
             {"role": "assistant", "parts": [
                 _tool("write", {"filePath": f"/repo/{i}.py", "content": "x=1"}, "ok")]},
             _text("assistant", "ok")]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out), balance=True)
    # largest bucket is file-edit (2); shell upsampled to match
    assert counts["file-edit"] == 2
    assert counts["shell"] == 2
    assert counts["balanced"] == 4
    assert (out / "train.balanced.jsonl").exists()


def test_emit_strata_balance_downsamples(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    # a long shell session -> many windows (max_turns=2), downsampled by balance
    shell_msgs = []
    for i in range(6):
        shell_msgs.append(_text("user", f"step {i}"))
        shell_msgs.append({"role": "assistant", "parts": [
            _tool("bash", {"command": f"echo {i}"}, "ok")]})
    bm.update(_write_session(cleaned / "s.json", "s1", "shell", shell_msgs))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/x.py", "content": "x=1"}, "ok")]},
         _text("assistant", "ok")]))
    cfg = make_cfg(
        paths={"raw_dir": str(tmp_root / "data" / "raw"),
               "cleaned_dir": str(cleaned),
               "dataset_dir": str(tmp_root / "data" / "datasets")},
        format={"max_turns_per_example": 2, "template": "chatml",
                "max_chars_per_example": 0})
    counts = emit_strata(cfg, bm, str(out), balance=True, cap=2)
    # shell had >2 windows but is capped down to 2; file-edit upsampled to 2
    assert counts["shell"] == 2
    assert counts["file-edit"] == 2
    assert counts["balanced"] == 4
