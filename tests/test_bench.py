"""Tests for src.bench (agentic task-completion harness).

All tests use a FakeDriver that returns scripted tool calls, so no GPU or
network is required. This exercises the parser, verifiers, sandbox tool env,
and the run-to-completion loop.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import bench as B


class FakeDriver(B.ModelDriver):
    """Returns a queue of canned assistant messages, then stops calling tools."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages, max_new_tokens=512):
        if self.replies:
            return self.replies.pop(0)
        return "Done, no more tools."


def test_parse_tool_calls_real_sep():
    text = ('<tool_call name="write" call_id="c1">'
            '{"filePath": "a.txt", "content": "hi"}'
            '\u276E\u276E\u276E')
    calls = B.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "write"
    assert calls[0]["args"]["filePath"] == "a.txt"


def test_parse_tool_calls_literal_escaped_sep():
    text = ('<tool_call name="bash">{"command": "ls"}'
            '\\u276E\\u276E\\u276E')
    calls = B.parse_tool_calls(text)
    assert calls[0]["name"] == "bash"
    assert calls[0]["args"]["command"] == "ls"


def test_parse_no_tool_call():
    assert B.parse_tool_calls("just talking, no tools") == []


def test_toolenv_write_read_edit(tmp_path):
    env = B.ToolEnv(tmp_path)
    r = env.execute("write", {"filePath": "x.txt", "content": "abc"})
    assert "wrote" in r
    r = env.execute("read", {"filePath": "x.txt"})
    assert r == "abc"
    r = env.execute("edit", {"filePath": "x.txt", "oldText": "abc", "newText": "abd"})
    assert "edited" in r
    assert env.execute("read", {"filePath": "x.txt"}) == "abd"


def test_toolenv_bash(tmp_path):
    env = B.ToolEnv(tmp_path)
    out = env.execute("bash", {"command": "echo hello-world"})
    assert "hello-world" in out


def test_verify_checks(tmp_path):
    (tmp_path / "f.txt").write_text("the quick brown fox")
    p, t, d = B.verify_task(tmp_path, [
        {"kind": "file_exists", "path": "f.txt"},
        {"kind": "file_contains", "path": "f.txt", "expect": "brown"},
        {"kind": "file_regex", "path": "f.txt", "pattern": r"q\w+ck"},
        {"kind": "command_exit", "cmd": "true", "expect_code": 0},
        {"kind": "command_output", "cmd": "echo zap", "expect": "zap"},
        {"kind": "file_exists", "path": "missing.txt"},
    ])
    assert (p, t) == (5, 6)


def test_run_task_success(tmp_path):
    driver = FakeDriver([
        ('<tool_call name="write" call_id="1">'
         '{"filePath": "greeting.txt", "content": "Hello, agent."}'
         '\u276E\u276E\u276E'),
        "All done.",
    ])
    task = B.Task.from_dict(json.loads(
        '{"id":"t1","prompt":"make the file",'
        '"checks":[{"kind":"file_exists","path":"greeting.txt"},'
        '{"kind":"file_contains","path":"greeting.txt","expect":"Hello, agent."}]}'))
    res = B.run_task(driver, task, model_name="fake", runner_name="self",
                     sandbox_root=tmp_path)
    assert res.success
    assert res.completed
    assert res.checks_passed == 2 and res.checks_total == 2


def test_run_task_failure(tmp_path):
    # model claims done without ever creating the file
    driver = FakeDriver(["I'm finished without doing anything."])
    task = B.Task.from_dict(json.loads(
        '{"id":"t2","prompt":"make the file",'
        '"checks":[{"kind":"file_exists","path":"greeting.txt"}]}'))
    res = B.run_task(driver, task, model_name="fake", runner_name="self",
                     sandbox_root=tmp_path)
    assert not res.success
    assert res.checks_passed == 0


def test_run_task_max_turns(tmp_path):
    # loops forever calling a tool; should stop at max_turns
    driver = FakeDriver([
        '<tool_call name="bash" call_id="1">{"command":"echo x"}\u276E\u276E\u276E'
        for _ in range(20)
    ])
    task = B.Task.from_dict(json.loads(
        '{"id":"t3","prompt":"loop","max_turns":3,'
        '"checks":[{"kind":"command_exit","cmd":"true","expect_code":0}]}'))
    res = B.run_task(driver, task, model_name="fake", runner_name="self",
                     sandbox_root=tmp_path)
    assert res.turns == 3
    assert res.error == "max turns reached without completion"


def test_load_tasks():
    p = Path(__file__).parent.parent / "eval" / "tasks" / "sample.jsonl"
    tasks = B.load_tasks(str(p))
    assert len(tasks) >= 4
    assert any(t.kind == "replay" for t in tasks)


def test_make_driver_unknown():
    try:
        B.make_driver("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_format_bench_results():
    r1 = B.TaskResult("a", "exec", "m", "self", checks_passed=2, checks_total=2,
                      turns=1, completed=True)
    r2 = B.TaskResult("b", "exec", "m", "self", checks_passed=0, checks_total=1,
                      turns=3, completed=False)
    md = B.format_bench_results([r1, r2])
    assert "1/2 tasks" in md
    assert "| a |" in md and "| b |" in md


def test_cli_bench_runs_fake(monkeypatch):
    import sys, importlib
    cli = importlib.import_module("src.cli")
    B2 = importlib.import_module("src.bench")

    class FakeDriver(B2.ModelDriver):
        def __init__(self): self.n = 0
        def generate(self, messages, max_new_tokens=512):
            self.n += 1
            if self.n == 1:
                return ('<tool_call name="write" call_id="1">'
                        '{"filePath":"g.txt","content":"Hello, agent."}'
                        '\u276E\u276E\u276E')
            return "done"

    tasks = [B2.Task.from_dict(json.loads(
        '{"id":"t1","prompt":"x",'
        '"checks":[{"kind":"file_exists","path":"g.txt"},'
        '{"kind":"file_contains","path":"g.txt","expect":"Hello, agent."}]}'))]
    monkeypatch.setattr(B2, "load_tasks", lambda p: tasks)
    monkeypatch.setattr(B2, "make_driver", lambda runner, **kw: FakeDriver())

    rc = cli.main(["bench", "--runner=self", "--model=/tmp/fake",
                   "--tasks=/tmp/none"])
    assert rc == 0


def test_hermes_driver_run_one(monkeypatch, tmp_path):
    import importlib
    B2 = importlib.import_module("src.bench")
    traj = {"completed": True, "api_calls": 4, "conversations": [1, 2]}

    class FakeHermes(B2.HermesDriver):
        def run_one(self, prompt, out_file):
            Path(out_file).write_text(json.dumps(traj))
            return traj

    monkeypatch.setattr(B2, "HermesDriver", FakeHermes)
    drv = B2.make_driver("hermes", hermes_dir="/tmp/x")
    assert isinstance(drv, B2.HermesDriver)
    # run_task with a submit-style driver trusts Hermes's completed flag
    task = B2.Task.from_dict(json.loads('{"id":"h1","prompt":"do it"}'))
    res = B2.run_task(drv, task, "hermes", "hermes")
    assert res.completed is True
    assert res.success is True
    assert res.checks_passed == 1

