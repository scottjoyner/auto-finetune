"""Offline guardian regression: no Signal requests or real GPU tests."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import finetune_guardian as g


def test_identifies_trainers_not_harness_or_shells():
    for args in [
        ["/media/venv/bin/python3", "-m", "src.cli", "train", "--source=hermes"],
        ["/bin/python3", "-u", "-m", "src.cli", "dpo", "--dry-run"],
        ["python3", "-m", "src.train", "--label=foo"],
        ["python", "-m", "src.dpo", "--label=bar"],
    ]:
        assert g.is_training_command(args), args
    for args in [
        ["/media/venv/bin/python3", "-m", "src.lfm_harness", "--base-url", "http://127.0.0.1"],
        ["/bin/python", "-m", "src.cli", "scheduler-run"],
        ["bash", "-c", "python -m src.cli train"],
        ["/usr/bin/pgrep", "-af", "src.cli train"],
        ["python3", "-m", "src.cli", "train--not-a-command"],
        ["python3", "-m", "src.cli", "--help"],
        [], ["python3"], ["python3", "-m"],
    ]:
        assert not g.is_training_command(args), args


def test_real_proc_cmdlines_only(tmp_path):
    for pid, argv in [
        ("111", ["python3", "-m", "src.lfm_harness", "--base-url", "local"]),
        ("112", ["python3", "-m", "src.cli", "train"]),
        ("113", ["bash", "-c", "python3 -m src.cli train"]),
        ("114", ["python3", "-m", "src.cli", "dpo"]),
    ]:
        path = tmp_path / pid
        path.mkdir()
        (path / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
    assert g.count_trainers(tmp_path) == 2


def test_held_idle_never_reports_trainer_death():
    previous = {"trainers_active": False, "held": True, "hot_streak": 0, "latched": []}
    for _ in range(12):
        previous, events = g.decide(previous, True, 0, 67)
        assert not events
        assert not previous["trainers_active"]


def test_first_idle_without_prior_run_no_death_event():
    state, events = g.decide({}, False, 0, 61)
    assert events == []
    assert not state["trainers_active"]


def test_real_trainer_exits_alerts_once_only():
    state, events = g.decide({}, False, 1, 64)
    assert events == []
    state, events = g.decide(state, False, 0, 62)
    assert events == ["trainer_disappeared"]
    state, events = g.decide(state, False, 0, 62)
    assert events == []


def test_held_trainer_detected_once_and_rearms_after_clear():
    state, events = g.decide({}, True, 1, 70)
    assert events == ["trainer_running_during_hold"]
    state, events = g.decide(state, True, 1, 69)
    assert events == []
    state, events = g.decide(state, True, 0, 70)
    assert events == []
    _, events = g.decide(state, True, 1, 70)
    assert events == ["trainer_running_during_hold"]


def test_two_consecutive_temperature_breaches_then_dedupe():
    state, alerts = g.decide({}, True, 0, 88)
    assert alerts == []
    state, alerts = g.decide(state, True, 0, 89)
    assert alerts == ["sustained_temperature"]
    state, alerts = g.decide(state, True, 0, 90)
    assert alerts == []
    state, alerts = g.decide(state, True, 0, 65)
    assert alerts == []
    state, alerts = g.decide(state, True, 0, 88)
    assert alerts == []
    _, alerts = g.decide(state, True, 0, 88)
    assert alerts == ["sustained_temperature"]


def test_unknown_gpu_temp_never_zero_fabricated_heat():
    state, _ = g.decide({}, True, 0, 89)
    state, events = g.decide(state, True, 0, None)
    assert state["hot_streak"] == 0
    assert events == []


def test_atomic_receipt_permissions_and_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "STATE", tmp_path / "state" / "guardian-state.json")
    payload = {"held": True, "trainers_active": False,
               "hot_streak": 0, "latched": []}
    g._atomic_state(payload)
    assert g.STATE.stat().st_mode & 0o077 == 0
    assert g.STATE.parent.stat().st_mode & 0o077 == 0
    assert g._read_state() == payload


def test_live_dry_run_cannot_send_or_mutate(tmp_path, monkeypatch, capsys):
    stop = tmp_path / "stop"
    stop.write_text("held")
    monkeypatch.setattr(g, "STOP", stop)
    monkeypatch.setattr(g, "STATE", tmp_path / "fresh.json")
    monkeypatch.setattr(g, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(g, "count_trainers", lambda: 0)
    monkeypatch.setattr(g, "temperature", lambda: 65)
    def prohibit(*args, **kwargs):
        raise AssertionError("dry-run sent notification")
    monkeypatch.setattr(g, "_send_event", prohibit)
    monkeypatch.setattr(g, "_atomic_state", prohibit)
    monkeypatch.setattr(g, "_append_log", prohibit)
    monkeypatch.setattr("sys.argv", ["finetune_guardian.py", "--dry-run"])
    assert g.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hold"] is True
    assert result["trainers"] == 0
    assert result["alerts"] == []


def test_notification_json_escapes_not_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "SIGNAL", "http://127.0.0.1:18080/api/v1/rpc")
    requests = []

    class FakeResp:
        status = 200

        def __enter__(self): return self

        def __exit__(self, *args): return None

    class FakeOpener:
        def open(self, req, timeout):
            requests.append(req)
            return FakeResp()

    monkeypatch.setattr(g, "build_opener", lambda *args: FakeOpener())
    assert g._send_event("trainer_disappeared", "2026-10-23T05:00:00Z", 70)
    payload = json.loads(requests[0].data)
    assert payload["method"] == "send"
    assert payload["params"]["noteToSelf"] is True
    assert requests[0].full_url == "http://127.0.0.1:18080/api/v1/rpc"
