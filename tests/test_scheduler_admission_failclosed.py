"""Synthetic admission and scheduler honesty tests; no live workloads."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src import scheduled_retrain as scheduled
from src import realtime_pipeline as realtime


def scheduler():
    instance = object.__new__(scheduled.ScheduledRetrainer)
    instance.schedule = scheduled.ScheduleConfig(require_gpu_free=True)
    instance.state = scheduled.ScheduleState()
    instance.repo = "/synthetic/no-runtime"
    instance._save_state = lambda: None
    instance._should_run = lambda: (True, "synthetic")
    instance.alerts = SimpleNamespace(fire=lambda *args, **kwargs: None)
    return instance


@pytest.mark.parametrize("error", [False, True])
def test_scheduler_does_not_admit_unknown_gpu_ownership(monkeypatch, error):
    instance = scheduler()

    def census(*args, **kwargs):
        if error:
            raise PermissionError("unavailable")
        return [{"pid": 99}]
    monkeypatch.setattr(scheduled, "unmanaged_training_processes", census)
    monkeypatch.setattr(
        scheduled.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("pgrep reached after ownership HOLD"),
    )
    assert instance._gpu_available() is False


@pytest.mark.parametrize("exitcode", [0, 1, 2, 127])
def test_scheduler_pgrep_codes_fail_closed(monkeypatch, exitcode):
    instance = scheduler()
    monkeypatch.setattr(scheduled, "unmanaged_training_processes", lambda *a: [])
    monkeypatch.setattr(
        scheduled.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], exitcode, ""),
    )
    assert instance._gpu_available() is (exitcode == 1)


def test_scheduler_command_errors_fail_closed(monkeypatch):
    instance = scheduler()
    monkeypatch.setattr(scheduled, "unmanaged_training_processes", lambda *a: [])
    monkeypatch.setattr(
        scheduled.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    assert instance._gpu_available() is False


def test_scheduler_exit_78_not_counted_as_success(monkeypatch):
    instance = scheduler()
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED", "1")
    seen = []

    def synthetic_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        if kwargs.get("check") and cmd:
            raise subprocess.CalledProcessError(78, cmd)
        return subprocess.CompletedProcess(cmd, 78)

    monkeypatch.setattr(scheduled.subprocess, "run", synthetic_run)
    result = instance.run()
    assert result["status"] == "failed"
    assert instance.state.runs_completed == 0
    assert instance.state.runs_failed == 1
    assert instance.state.current_status == "error"
    assert len(seen) == 1
    assert seen[0][1]["check"] is True
    assert "AF_SKIP_UNMANAGED_CHECK" not in seen[0][1]["env"]
    assert "TRAIN_ALLOW_UNVERIFIED" not in seen[0][1]["env"]


def test_realtime_child_command_removes_inherited_flags(monkeypatch):
    instance = object.__new__(realtime.RealtimePipeline)
    instance.repo = "/synthetic/no-runtime"
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED", "1")
    captured = []

    def synthetic_run(cmd, **kwargs):
        captured.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(realtime.subprocess, "run", synthetic_run)
    rc, text = instance._run_cmd(["python", "-m", "src.cli", "train"],
                                 extra_env={"AF_SKIP_UNMANAGED_CHECK": "1",
                                            "TRAIN_ALLOW_UNVERIFIED": "1"})
    assert rc == 0 and text == "ok"
    assert captured
    assert "AF_SKIP_UNMANAGED_CHECK" not in captured[0]["env"]
    assert "TRAIN_ALLOW_UNVERIFIED" not in captured[0]["env"]
