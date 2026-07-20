"""Tests for the mined-task benchmark packaging (bench.build_auto_bench)
and the hardened sandbox/tool loop (bench.ToolEnv / run_task)."""
import json
import tempfile
from pathlib import Path

from src.bench import (
    ModelDriver,
    Task,
    ToolEnv,
    build_auto_bench,
    load_tasks,
    run_task,
)


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def test_build_auto_bench_filters_unverified(tmp_path: Path) -> None:
    at = tmp_path / "auto-tasks.jsonl"
    vr = tmp_path / "verify-report.jsonl"
    at.write_text("\n".join([
        json.dumps({"task_id": "t-ok", "instruction": "make a file", "kind": "exec",
                    "checks": [{"kind": "file_contains", "path": "a.txt", "expect": "x"}],
                    "bucket": "debug", "difficulty": "easy", "source": "hermes"}),
        json.dumps({"task_id": "t-bad", "instruction": "make b", "kind": "exec",
                    "checks": [{"kind": "file_contains", "path": "b.txt", "expect": "y"}]}),
        # absolute-path check: can't be reproduced in a fresh sandbox -> dropped
        json.dumps({"task_id": "t-abs", "instruction": "make abs", "kind": "exec",
                    "checks": [{"kind": "file_contains", "path": "/home/scott/x", "expect": "z"}],
                    "auto": True}),
    ]) + "\n")
    vr.write_text("\n".join([
        json.dumps({"task_id": "t-ok", "ok": True, "reason": "ok"}),
        json.dumps({"task_id": "t-bad", "ok": False, "reason": "content mismatch"}),
    ]) + "\n")

    out = tmp_path / "auto-verified.jsonl"
    n = build_auto_bench(str(at), str(vr), str(out))
    assert n == 1, f"expected 1 verifiable task, got {n}"

    tasks = load_tasks(str(out))
    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "t-ok"
    assert t.kind == "exec"
    assert t.prompt == "make a file"
    assert t.checks[0]["path"] == "a.txt"
    # carries metadata for reporting
    assert t.bucket == "debug"


def test_build_auto_bench_without_verify_report(tmp_path: Path) -> None:
    at = tmp_path / "auto-tasks.jsonl"
    at.write_text(json.dumps({
        "task_id": "t1", "instruction": "i1", "kind": "exec",
        "checks": [{"kind": "file_contains", "path": "f.txt", "expect": "v"}],
    }) + "\n")
    out = tmp_path / "out.jsonl"
    # no verify report -> keeps all relative file_contains tasks
    n = build_auto_bench(str(at), None, str(out), only_verified=False)
    assert n == 1
    assert load_tasks(str(out))[0].id == "t1"


def test_toolenv_bash_is_sandboxed() -> None:
    with tempfile.TemporaryDirectory() as d:
        env = ToolEnv(Path(d))
        blocked = env.execute("bash", {"command": "sudo rm -rf /"})
        assert "blocked" in blocked, blocked
        # benign command still runs
        ok = env.execute("bash", {"command": "echo hello"})
        assert "hello" in ok, ok


def test_run_task_end_to_end_with_fake_driver(tmp_path: Path) -> None:
    # a model that emits exactly the write needed to satisfy the check
    class FakeDriver(ModelDriver):
        def __init__(self, calls):
            self.calls = list(calls)
            self._i = 0

        def generate(self, messages, max_new_tokens=512):
            if self._i < len(self.calls):
                name, args = self.calls[self._i]
                self._i += 1
                obj = {"name": name, "arguments": args}
                return f"<tool_call>\n{json.dumps(obj)}\n</tool_call>"
            return "task complete"

    task = Task(id="t1", kind="exec", prompt="write out.txt with hello",
                checks=[{"kind": "file_contains", "path": "out.txt", "expect": "hello"}],
                tools=["write"], max_turns=4)
    driver = FakeDriver([("write", {"filePath": "out.txt", "content": "hello"})])
    res = run_task(driver, task, "fake", "fake", sandbox_root=tmp_path)
    assert res.success, res.transcript
    assert res.completed
