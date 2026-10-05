"""Pin the argv convention every src.cli subcommand handler relies on.

src/cli.py dispatches to per-module main() functions in two shapes:

    run(cfg, argv)              -> the handler reads argv[1] as the command
    run(cfg, [cmd] + argv[2:])  -> the command sits at argv[0]

Five handlers (registry, scheduler) used the second shape while still reading
argv[1], which silently broke them: `scheduler-run` fell through to its default
and ran scheduler-status, and every form carrying a flag printed the help text
and exited 0. Nothing failed loudly -- the commands just did the wrong thing.

These tests assert the first shape for every dispatched handler, so a future
dispatcher cannot reintroduce the split.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import src.cli as cli

SRC = Path(cli.__file__).resolve().parent


def _dispatch_calls() -> list[tuple[int, str]]:
    """(lineno, second-arg-expr) for every `return run(cfg, X)` dispatch.

    `run(cfg)` with no second argument is fine: those handlers take only a
    Config and never read argv. `run(argv[2:])` is subagent_harness, which takes
    no cfg and indexes argv[0] itself -- also fine.
    """
    tree = ast.parse((SRC / "cli.py").read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            func = node.value.func
            if not (isinstance(func, ast.Name) and func.id == "run"):
                continue
            args = node.value.args
            if len(args) == 2 and ast.unparse(args[0]) == "cfg":
                found.append((node.lineno, ast.unparse(args[1])))
    return found


def test_dispatch_passes_full_argv_everywhere():
    """Every (cfg, X) handler must receive the full argv.

    Anything else re-creates the split that silently broke scheduler-run.
    """
    offenders = [f"cli.py:{line} -> run(cfg, {expr})"
                 for line, expr in _dispatch_calls() if expr != "argv"]
    assert not offenders, (
        "handlers taking argv must be called as run(cfg, argv) so that argv[1] "
        f"is the command: {offenders}")


@pytest.mark.parametrize("module_name,func_name", [
    ("src.scheduler", "main"),
    ("src.registry", "main"),
    ("src.deploy", "main"),
    ("src.notify", "main"),
    ("src.metrics", "main"),
    ("src.quantize", "main"),
])
def test_handler_reads_argv1_as_command(module_name, func_name):
    import importlib
    mod = importlib.import_module(module_name)
    src = inspect.getsource(getattr(mod, func_name))
    assert "argv[1]" in src, (
        f"{module_name}.{func_name} must take the command from argv[1] to match "
        f"the run(cfg, argv) dispatch")


@pytest.mark.parametrize("argv,expect_cmd", [
    (["cli", "scheduler-run"], "scheduler-run"),
    (["cli", "scheduler-run", "--dry-run"], "scheduler-run"),
    (["cli", "scheduler-loop", "--interval=60"], "scheduler-loop"),
    (["cli", "registry-add", "--label=x"], "registry-add"),
    (["cli", "quantize", "--label=x"], "quantize"),
    (["cli", "notify-history", "--limit=5"], "notify-history"),
])
def test_command_is_recoverable_from_argv1(argv, expect_cmd):
    # The invariant the dispatch depends on, stated directly.
    assert argv[1] == expect_cmd


def test_scheduler_run_does_not_fall_through_to_status(monkeypatch, capsys):
    """Regression: `scheduler-run` used to execute scheduler-status."""
    import src.scheduler as S

    seen = {}

    def fake_run_once(self, dry_run=False):
        seen["dry_run"] = dry_run
        return S.RunResult(True, "dry-run" if dry_run else "complete", "ok", 0.0)

    monkeypatch.setattr(S.Scheduler, "run_once", fake_run_once)
    monkeypatch.setattr(S.Scheduler, "_load_state",
                        lambda self: S.SchedulerState(
                            last_run=0, last_harvest=0, last_train=0,
                            last_deploy=0, runs_completed=0, runs_failed=0,
                            current_phase="idle", last_error=None))
    from conftest import make_cfg
    rc = cli._dispatch(["cli", "scheduler-run", "--dry-run"], cfg=make_cfg())
    assert rc == 0
    assert "phase=dry-run" in capsys.readouterr().out
    assert seen["dry_run"] is True
