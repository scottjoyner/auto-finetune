"""Fail-closed, disposable scheduling tests without GPU, Neo4j, or training."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import scheduled_retrain as scheduler
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path)}})


def make(tmp_path,**schedule_kwargs):
    opts=dict(interval_hours=0,require_gpu_free=False,require_data_fresh=False,
              notify_on_train=False,notify_on_skip=False)
    opts.update(schedule_kwargs)
    return scheduler.ScheduledRetrainer(cfg(tmp_path),scheduler.ScheduleConfig(**opts))


def test_scheduler_replays_persistent_state_and_ignores_unknown_fields(tmp_path):
    r=make(tmp_path)
    r.state.last_train=22.5
    r.state.runs_completed=2
    r._save_state()
    disk=json.loads((tmp_path/"scheduled-retrain-state.json").read_text())
    disk["future_unknown_option"]="ignored"
    (tmp_path/"scheduled-retrain-state.json").write_text(json.dumps(disk))
    replay=make(tmp_path)
    assert replay.state.runs_completed==2
    assert replay.state.last_train==22.5


def test_corrupt_schedule_state_is_not_promoted_to_a_valid_run(tmp_path):
    (tmp_path/"scheduled-retrain-state.json").write_text("{truncated")
    assert make(tmp_path).state.runs_completed==0


def test_check_interval_prioritizes_cooldown(tmp_path,monkeypatch):
    r=make(tmp_path,interval_hours=6)
    monkeypatch.setattr(scheduler.time,"time",lambda:10000.0)
    r.state.last_run=10000
    assert r._should_run()[0] is False
    assert "Interval not elapsed" in r._should_run()[1]


def test_resource_denial_and_failure_latch(tmp_path,monkeypatch):
    r=make(tmp_path,require_gpu_free=True)
    monkeypatch.setattr(r,"_gpu_available",lambda:False)
    assert r._should_run()==(False,"GPU not available (training in progress)")
    monkeypatch.setattr(r,"_gpu_available",lambda:True)
    r.state.consecutive_failures=3
    assert not r._should_run()[0]


def test_critical_staleness_does_not_trigger_train(tmp_path,monkeypatch):
    r=make(tmp_path,require_data_fresh=True)
    monkeypatch.setattr(scheduler,"assess_freshness",
                        lambda cfg:SimpleNamespace(overall_staleness="critical"))
    assert r._should_run()==(False,"Data critically stale")


def test_skip_never_launches_subprocess(tmp_path,monkeypatch):
    r=make(tmp_path,notify_on_skip=True)
    monkeypatch.setattr(r,"_should_run",lambda:(False,"no fresh data"))
    monkeypatch.setattr(scheduler.subprocess,"run",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("pipeline must not execute")))
    alerts=[]
    monkeypatch.setattr(r.alerts,"fire",lambda *a,**k: alerts.append(a))
    result=r.run()
    assert result["status"]=="skipped"
    assert r.state.runs_skipped==1
    assert r.state.runs_completed==0
    assert len(alerts)==1
    assert make(tmp_path).state.runs_completed==0


def test_successful_disposable_pipeline_checks_every_stage_and_strips_bypass(tmp_path,monkeypatch):
    r=make(tmp_path,notify_on_train=True)
    monkeypatch.setattr(r,"_should_run",lambda:(True,"OK"))
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK","1")
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED","1")
    recorded=[]
    def fake_run(cmd,**kwargs):
        recorded.append((cmd,kwargs))
        assert kwargs["check"] is True
        assert kwargs["timeout"] in (300,600)
        assert "AF_SKIP_UNMANAGED_CHECK" not in kwargs["env"]
        assert "TRAIN_ALLOW_UNVERIFIED" not in kwargs["env"]
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(scheduler.subprocess,"run",fake_run)
    calls=[]
    monkeypatch.setattr(r.alerts,"fire",lambda *a,**k: calls.append(a))
    outcome=r.run()
    assert outcome["status"]=="completed"
    assert r.state.runs_completed==1
    assert r.state.runs_failed==0
    assert len(recorded)==4
    assert len(calls)==1
    assert "neo4j-traces" in recorded[0][0]
    assert "realtime-pipeline" in recorded[-1][0]
    assert make(tmp_path).state.runs_completed==1


@pytest.mark.parametrize("failing_stage",[0,1,2,3])
def test_any_nonzero_subprocess_exit_blocks_success(tmp_path,monkeypatch,failing_stage):
    r=make(tmp_path,notify_on_train=True)
    monkeypatch.setattr(r,"_should_run",lambda:(True,"OK"))
    count=[]
    def fake_run(cmd,**kwargs):
        assert kwargs["check"] is True
        count.append(cmd)
        if len(count)-1==failing_stage:
            raise subprocess.CalledProcessError(13,cmd)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(scheduler.subprocess,"run",fake_run)
    errors=[]
    monkeypatch.setattr(r.alerts,"fire",lambda *a,**k: errors.append(a))
    status=r.run()
    assert status["status"]=="failed"
    assert r.state.runs_completed==0
    assert r.state.runs_failed==1
    assert r.state.consecutive_failures==1
    assert "13" in r.state.last_error
    assert len(count)==failing_stage+1
    assert make(tmp_path).state.runs_completed==0
    assert len(errors)==1


def test_scheduler_gpu_guard_respects_unmanaged_trainer(tmp_path,monkeypatch):
    r=make(tmp_path,require_gpu_free=True)
    monkeypatch.setattr(scheduler,"unmanaged_training_processes",lambda _:["synthetic PID"])
    monkeypatch.setattr(scheduler.subprocess,"run",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("must not probe if unmanaged")))
    assert r._gpu_available() is False


def test_cli_run_reports_disposable_skip_status(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(scheduler,"load",lambda:cfg(tmp_path))
    monkeypatch.setattr(scheduler.ScheduledRetrainer,"run",
                        lambda self:{"status":"skipped","reason":"fixture"})
    assert scheduler.main(["--run","--interval","1"])==0
    assert '"status": "skipped"' in capsys.readouterr().out


def test_gpu_guard_denies_pgrep_resident_trainer(tmp_path,monkeypatch):
    r=make(tmp_path,require_gpu_free=True)
    monkeypatch.setattr(scheduler,"unmanaged_training_processes",lambda _:[])
    monkeypatch.setattr(scheduler.subprocess,"run",
                        lambda *a,**k:SimpleNamespace(returncode=0,stdout="synthetic pid"))
    assert r._gpu_available() is False


def test_gpu_guard_accepts_idle_subprocess_monitor(tmp_path,monkeypatch):
    r=make(tmp_path,require_gpu_free=True)
    monkeypatch.setattr(scheduler,"unmanaged_training_processes",lambda _:[])
    monkeypatch.setattr(scheduler.subprocess,"run",
                        lambda *a,**k:SimpleNamespace(returncode=1,stdout=""))
    assert r._gpu_available() is True


def test_loop_keyboard_interrupt_resets_disposable_state(tmp_path,monkeypatch):
    r=make(tmp_path)
    monkeypatch.setattr(r,"run",lambda:{"status":"skipped"})
    monkeypatch.setattr(scheduler.time,"sleep",
                        lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    r.state.current_status="running"
    r.run_loop(check_interval=1)
    assert r.state.current_status=="idle"
    assert make(tmp_path).state.current_status=="idle"


def test_scheduler_cli_loop_uses_local_mock_without_sleep(tmp_path,monkeypatch):
    monkeypatch.setattr(scheduler,"load",lambda:cfg(tmp_path))
    seen=[]
    monkeypatch.setattr(scheduler.ScheduledRetrainer,"run_loop",
                        lambda self,check_interval:seen.append(check_interval))
    assert scheduler.main(["--loop","--check-interval","7"])==0
    assert seen==[7]
