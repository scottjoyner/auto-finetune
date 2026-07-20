"""Tests for src.bench (agentic task-completion harness).

All tests use a FakeDriver that returns scripted tool calls, so no GPU or
network is required. This exercises the parser, verifiers, sandbox tool env,
and the run-to-completion loop.
"""
from __future__ import annotations

import importlib
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
    import importlib
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


def test_parse_canonical_base_format():
    B = importlib.import_module("src.bench")
    text = ('planning...\n<tool_call>\n'
            '{"name": "bash", "arguments": {"command": "ls -la"}}\n'
            '</tool_call>')
    calls = B.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"
    assert calls[0]["args"]["command"] == "ls -la"


def test_format_tool_result_variants():
    B = importlib.import_module("src.bench")
    assert B.format_tool_result("x", "base") == "<tool_response>\nx\n</tool_response>"
    assert B.format_tool_result("x", "finetune") == "<tool_result>x</tool_result>"


def test_optimized_driver_variant_autodetect():
    import importlib
    B = importlib.import_module("src.bench")

    class FakeOpt(B.OptimizedDriver):
        def _load(self):
            self._model = object()
            self._tok = None
    # base path -> variant 'base'
    d = FakeOpt("/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b")
    assert d.variant == "base"
    assert d.wrap_result("o") == "<tool_response>\no\n</tool_response>"
    # finetune path -> variant 'finetune'
    d2 = FakeOpt("/media/scott/data/.../toolcall-v5-3b-ssd-merged")
    assert d2.variant == "finetune"
    assert d2.wrap_result("o") == "<tool_result>o</tool_result>"


def test_run_task_uses_wrap_result_for_subagent(tmp_path):
    import importlib
    B = importlib.import_module("src.bench")

    class FakeOpt(B.OptimizedDriver):
        def __init__(self):
            self.variant = "base"
        def generate(self, messages, max_new_tokens=512):
            # emit a canonical base-format call
            return ('<tool_call>\n{"name":"write",'
                    '"arguments":{"filePath":"g.txt","content":"hi"}}\n</tool_call>')
        def wrap_result(self, r):
            return B.format_tool_result(r, self.variant)

    task = B.Task.from_dict(json.loads(
        '{"id":"t9","prompt":"x",'
        '"checks":[{"kind":"file_exists","path":"g.txt"},'
        '{"kind":"file_contains","path":"g.txt","expect":"hi"}]}'))
    res = B.run_task(FakeOpt(), task, "opt", "subagent", sandbox_root=tmp_path)
    assert res.success
    # the recovery/error path is exercised when args are None
    assert res.checks_passed == 2


def test_run_task_error_recovery_feedbacks_message(tmp_path):
    # When a tool call has unparseable args (args is None), the harness feeds a
    # structured ERROR back; a self-correcting model can then re-emit valid JSON.
    # We assert the loop surfaces the error and that a follow-up valid call works.
    class RecoveringDriver(B.ModelDriver):
        def __init__(self):
            self.calls = 0
        def generate(self, messages, max_new_tokens=512):
            self.calls += 1
            if self.calls == 1:
                # malformed: no valid JSON args -> parse yields args=None
                return '<tool_call name="write" call_id="1">{bad json}\u276E\u276E\u276E'
            if self.calls == 2:
                # recover with a valid call
                return ('<tool_call name="write" call_id="2">'
                        '{"filePath":"g.txt","content":"hi"}\u276E\u276E\u276E')
            return "done"
    task = B.Task.from_dict(json.loads(
        '{"id":"t10","prompt":"x",'
        '"checks":[{"kind":"file_exists","path":"g.txt"},'
        '{"kind":"file_contains","path":"g.txt","expect":"hi"}]}'))
    res = B.run_task(RecoveringDriver(), task, "m", "self", sandbox_root=tmp_path)
    assert res.success, res.transcript
    # the ERROR recovery message must have been sent back to the model
    tool_msgs = [t for t in res.transcript if t.get("role") == "tool"]
    assert any("ERROR" in (t.get("content") or "") for t in tool_msgs)
    assert res.turns == 3  # malformed + recovered + done


def test_verify_command_output_failure_detail(tmp_path):
    # command_output expect missing -> failed, but detail explains
    p, t, d = B.verify_task(tmp_path, [
        {"kind": "command_output", "cmd": "echo yes", "expect": "NOPE"},
        {"kind": "command_exit", "cmd": "false", "expect_code": 0},
    ])
    assert (p, t) == (0, 2)
    assert any("command_output" in x["detail"] for x in d)
    assert any("command_exit" in x["detail"] for x in d)


def test_run_task_replay_context_prepended(tmp_path):
    # replay tasks prepend replay_context as prior messages; ensure the loop
    # still runs and verifies the end-state.
    driver = FakeDriver([
        ('<tool_call name="write" call_id="1">'
         '{"filePath":"g.txt","content":"hi"}\u276E\u276E\u276E'),
        "done",
    ])
    task = B.Task.from_dict(json.loads(
        '{"id":"t11","kind":"replay","prompt":"continue",'
        '"replay_context":[{"role":"user","content":"prior context"}],'
        '"checks":[{"kind":"file_exists","path":"g.txt"}]}'))
    res = B.run_task(driver, task, "m", "self", sandbox_root=tmp_path)
    assert res.success
    # first assistant turn message should follow the replay context
    assert res.transcript[0]["role"] == "assistant"

