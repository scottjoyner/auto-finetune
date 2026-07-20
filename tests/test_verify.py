"""Tests for src.verify (safe replay of mined benchmark tasks)."""
from __future__ import annotations

import json

from src.verify import _parse_task_id, summarize, verify_all, verify_task


def _write_session(path, sid, messages):
    path.write_text(json.dumps(
        {"source": "hermes", "session_id": sid, "messages": messages}))


def _tool(name, inp=None, out=None):
    p = {"type": "tool", "tool": name}
    if inp is not None:
        p["input"] = inp
    if out is not None:
        p["output"] = out
    return p


def _text(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def test_parse_task_id():
    assert _parse_task_id("auto-hermes-h9") == ("hermes", "h9")
    assert _parse_task_id("auto-opencode-ses_demo") == ("opencode", "ses_demo")
    assert _parse_task_id("weird") == ("", "weird")


def test_verify_task_replays_and_passes(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    _write_session(cleaned / "h.json", "h9", [
        _text("user", "create a helper that sums a list"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/repo/sum.py",
                    "content": "def sum(xs):\n    return sum(xs)"}, "ok")]},
    ])
    from src.clean import _dedup_by_session
    from src.format_dataset import iter_cleaned_records
    sessions = _dedup_by_session(iter_cleaned_records(str(cleaned)))

    task = {"task_id": "auto-hermes-h9", "bucket": "file-edit",
             "difficulty": "easy",
             "checks": [{"kind": "file_contains", "path": "sum.py",
                         "expect": "def sum(xs):"}]}
    res = verify_task(task, sessions)
    assert res["ok"] is True
    assert res["reason"] == "ok"
    assert "sum.py" in res["replayed"]


def test_verify_task_session_missing():
    task = {"task_id": "auto-hermes-missing", "checks": []}
    res = verify_task(task, {})
    assert res["ok"] is False
    assert res["reason"] == "source session not found"


def test_verify_all_and_summarize(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    _write_session(cleaned / "h.json", "h9", [
        _text("user", "create add module"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/repo/add.py",
                    "content": "def add(a,b):\n    return a+b"}, "ok")]},
    ])
    tasks = tmp_path / "auto-tasks.jsonl"
    tasks.write_text(json.dumps({
        "task_id": "auto-hermes-h9", "bucket": "file-edit",
        "difficulty": "easy",
        "checks": [{"kind": "file_contains", "path": "add.py",
                    "expect": "def add(a,b):"}]}) + "\n")
    results = verify_all(str(tasks), str(cleaned))
    assert len(results) == 1
    assert results[0]["ok"] is True
    s = summarize(results)
    assert s["n_tasks"] == 1
    assert s["checks_passed"] == 1
    assert s["pass_rate"] == 1.0
