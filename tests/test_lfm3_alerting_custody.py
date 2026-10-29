"""Test alert/acknowledgement persistence using tmp-only storage and no networks."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.alerting import AlertManager
from src.config import Config


def manager(tmp_path, monkeypatch, webhook=""):
    monkeypatch.setattr(subprocess, "run", Mock(return_value=None))
    return AlertManager(Config(raw={
        "paths": {"analysis_dir": str(tmp_path)},
        "notify": {"webhook_url": webhook},
    }))


def test_empty_alert_store(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    assert a.alerts == []
    assert a.recent() == []
    assert a.unacknowledged() == []
    assert not a.acknowledge("missing")


def test_alert_round_trip_across_restart(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    id_ = a.fire("warning", "lfm3", "Research gate blocked")
    b = manager(tmp_path, monkeypatch)
    assert len(b.alerts) == 1
    assert b.alerts[0].alert_id == id_
    assert b.alerts[0].level == "warning"
    assert b.alerts[0].message == "Research gate blocked"
    assert len(b.unacknowledged()) == 1


def test_acknowledge_replayed_latest_state(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    id_ = a.fire("critical", "promotion", "Blocked")
    assert a.acknowledge(id_)
    assert a.unacknowledged() == []
    b = manager(tmp_path, monkeypatch)
    assert len(b.alerts) == 1
    assert b.alerts[0].acknowledged is True
    assert b.unacknowledged() == []
    assert len((Path(tmp_path) / "alerts.jsonl").read_text().splitlines()) == 2


def test_duplicate_acknowledgement_does_not_revive_old_record(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    id_ = a.fire("warning", "promotion", "Reviewed")
    assert a.acknowledge(id_)
    assert a.acknowledge(id_)
    assert len(manager(tmp_path, monkeypatch).alerts) == 1


def test_acknowledge_only_matching_id(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    a.fire("info", "a", "first")
    two = a.fire("error", "b", "second")
    assert a.acknowledge(two)
    b = manager(tmp_path, monkeypatch)
    assert len(b.unacknowledged()) == 1
    assert b.unacknowledged()[0].message == "first"


def test_alerts_recent_level_and_limit(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    a.fire("info", "demo", "one")
    a.fire("warning", "demo", "two")
    a.fire("error", "demo", "three")
    assert len(a.recent(limit=2)) == 2
    assert [x.message for x in a.recent(level="warning")] == ["two"]


def test_alert_metadata_roundtrip(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    a.fire("warning", "ab", "hold", metadata={"reviewed": False})
    restored = manager(tmp_path, monkeypatch)
    assert restored.alerts[0].metadata == {"reviewed": False}


def test_alert_log_contains_message_not_external_io(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    a.fire("warning", "lfm3", "synthetic-only")
    text = (Path(tmp_path) / "alerts.log").read_text()
    assert "synthetic-only" in text
    assert "WARNING" in text


def test_alert_corrupt_log_does_not_throw(tmp_path, monkeypatch):
    (Path(tmp_path) / "alerts.jsonl").write_text("{broken")
    a = manager(tmp_path, monkeypatch)
    assert a.alerts == []


def test_webhook_is_mocked_and_payload_is_structured(tmp_path, monkeypatch):
    imported = []
    def fake_urlopen(request, timeout):
        imported.append((json.loads(request.data), timeout))
        return Mock()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    a = manager(tmp_path, monkeypatch, webhook="https://invalid.test")
    a.fire("warning", "auto_promote", "held", metadata={"source": "synthetic"})
    assert len(imported) == 1
    assert imported[0][0]["metadata"] == {"source": "synthetic"}
    assert imported[0][1] == 5


def test_webhook_failure_does_not_break_alert_log(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise ConnectionError("mocked offline webhook")
    monkeypatch.setattr("urllib.request.urlopen", failure)
    a = manager(tmp_path, monkeypatch, webhook="https://invalid.test")
    a.fire("error", "promoter", "offline-only")
    assert len(manager(tmp_path, monkeypatch).alerts) == 1


def test_desktop_notification_failure_does_not_break_alerts(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise OSError("no desktop")
    monkeypatch.setattr(subprocess, "run", failure)
    a = AlertManager(Config(raw={
        "paths": {"analysis_dir": str(tmp_path)}, "notify": {"webhook_url": ""},
    }))
    a.fire("warning", "lfm3", "hold")
    assert len(a.alerts) == 1


def test_alert_source_remains_untrusted_metadata(tmp_path, monkeypatch):
    a = manager(tmp_path, monkeypatch)
    a.fire("info", "lfm3", "synthetic", metadata={"verified": False})
    b = manager(tmp_path, monkeypatch)
    assert b.alerts[0].metadata["verified"] is False


def test_alert_ids_unique_at_frozen_clock(tmp_path, monkeypatch):
    monkeypatch.setattr("src.alerting.time.time", lambda: 1234567890)
    a = manager(tmp_path, monkeypatch)
    first = a.fire("warning", "a", "first")
    second = a.fire("warning", "b", "second")
    assert first != second
    assert len(manager(tmp_path, monkeypatch).alerts) == 2
