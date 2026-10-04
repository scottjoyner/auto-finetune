"""Tests for notification delivery, logging and history.

Notifications are how the operator learns a training run died, so the tests here
check that the *reported status* is honest and that history survives the kind of
damage an append-only log actually takes. Three defects were found: a torn line
made the entire history unreadable, the desktop channel reported OK when
notify-send failed, and an unregistered event silently rendered as info.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from src import notify as N
from src.config import Config


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "analysis"
    os.makedirs(str(d), exist_ok=True)
    return Config(raw={"paths": {"analysis_dir": str(d)}})


def _log_path(cfg):
    return os.path.join(cfg.path("analysis_dir"), "notifications",
                        "notifications.jsonl")


def _entry(event="training_failed", message="boom", ts=1000.0,
           severity="error"):
    return json.dumps({"event": event, "message": message, "timestamp": ts,
                       "severity": severity, "data": None})


# --- severity table --------------------------------------------------------

@pytest.mark.parametrize("event,severity", [
    ("training_complete", "info"),
    ("training_failed", "error"),
    ("deploy_failed", "error"),
    ("rollback_failed", "error"),
    ("harvest_failed", "error"),
    ("eval_complete", "info"),
    ("error", "error"),
])
def test_known_event_severities(event, severity):
    assert N.EVENT_SEVERITY[event] == severity


def test_every_failure_variant_is_registered():
    # A new *_failed event that nobody registered would render as info.
    missing = [e for e in N.EVENT_SEVERITY if e.endswith("_failed")]
    assert all(N.EVENT_SEVERITY[e] == "error" for e in missing)


def test_unknown_event_warns_on_stderr(cfg, capsys, monkeypatch):
    # Regression: "training_faild" (a typo) silently became severity info.
    monkeypatch.setattr(N, "send_desktop", lambda *a: False)
    N.send_notification(cfg, "training_faild", "it broke")
    err = capsys.readouterr().err
    assert "not in EVENT_SEVERITY" in err
    assert "training_faild" in err


def test_unknown_event_does_not_escalate(capsys, cfg, monkeypatch):
    # Non-escalating on purpose: alert fatigue would hide real errors.
    monkeypatch.setattr(N, "send_desktop", lambda *a: False)
    N.send_notification(cfg, "some_new_info_event", "hello")
    assert "warning" in capsys.readouterr().err


def test_known_event_does_not_warn(cfg, capsys, monkeypatch):
    monkeypatch.setattr(N, "send_desktop", lambda *a: False)
    N.send_notification(cfg, "training_failed", "boom")
    assert "EVENT_SEVERITY" not in capsys.readouterr().err


# --- desktop channel -------------------------------------------------------

class _Proc:
    def __init__(self, code):
        self.returncode = code
        self.stdout = b""
        self.stderr = b""


def test_desktop_reports_failure(monkeypatch):
    # Regression: the exit code was ignored, so a failed send reported OK.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(1))
    assert N.send_desktop("t", "m") is False


def test_desktop_reports_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(0))
    assert N.send_desktop("t", "m") is True


def test_desktop_missing_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("notify-send")
    monkeypatch.setattr(subprocess, "run", boom)
    assert N.send_desktop("t", "m") is False


def test_desktop_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("notify-send", 5)
    monkeypatch.setattr(subprocess, "run", boom)
    assert N.send_desktop("t", "m") is False


# --- log + history ---------------------------------------------------------

def test_history_tolerates_torn_line(cfg):
    # Regression: a crash between the write and the newline left a partial
    # record, and json.loads raised, making ALL history unreadable forever.
    path = _log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_entry(event="a", ts=1.0) + "\n")
        f.write('{"event":"b","message":"tor')      # torn
        f.write("\n")
        f.write(_entry(event="c", ts=3.0) + "\n")
    hist = N.get_notification_history(cfg, 50)
    assert [h["event"] for h in hist] == ["a", "c"]


def test_history_skips_blank_and_garbage_lines(cfg):
    path = _log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_entry(event="a", ts=1.0) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write("not json at all\n")
        f.write(_entry(event="b", ts=2.0) + "\n")
    assert [h["event"] for h in N.get_notification_history(cfg, 50)] == ["a", "b"]


def test_history_returns_newest(cfg):
    path = _log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(10):
            f.write(_entry(event=f"e{i}", ts=float(i)) + "\n")
    # Append-only log, so file order is chronological: the last `limit`
    # entries, not re-sorted by timestamp.
    hist = N.get_notification_history(cfg, 3)
    assert [h["event"] for h in hist] == ["e7", "e8", "e9"]
    assert [h["timestamp"] for h in hist] == [7.0, 8.0, 9.0]


def test_history_missing_file_is_empty(cfg):
    assert N.get_notification_history(cfg) == []


def test_history_limit_larger_than_file(cfg):
    path = _log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_entry(event="a", ts=1.0) + "\n")
    assert len(N.get_notification_history(cfg, 100)) == 1


def test_log_notification_appends(cfg):
    N.log_notification(N.Notification("training_failed", "m1", 1.0, "error"),
                       os.path.join(cfg.path("analysis_dir"), "notifications"))
    N.log_notification(N.Notification("training_complete", "m2", 2.0, "info"),
                       os.path.join(cfg.path("analysis_dir"), "notifications"))
    hist = N.get_notification_history(cfg)
    assert [h["event"] for h in hist] == ["training_failed", "training_complete"]


def test_notification_preserves_data(cfg):
    N.send_notification(cfg, "training_failed", "m", {"loss": 0.5})
    hist = N.get_notification_history(cfg)
    assert hist[0]["data"] == {"loss": 0.5}


# --- channel routing -------------------------------------------------------

def test_send_notification_reports_each_channel(cfg, monkeypatch):
    monkeypatch.setattr(N, "send_desktop", lambda *a: True)
    monkeypatch.setattr(N, "send_webhook", lambda *a, **k: False)
    monkeypatch.setattr(N, "send_emailsmtp", lambda *a, **k: False)
    cfg.raw.setdefault("notify", {})
    cfg.raw["notify"].update({"webhook_url": "http://x", "email_to": "a@b"})
    res = N.send_notification(cfg, "training_failed", "boom")
    assert res["desktop"] is True
    assert res["webhook"] is False
    assert res["email"] is False
    assert res["log"] is True


def test_send_notification_skips_unconfigured_channels(cfg, monkeypatch):
    monkeypatch.setattr(N, "send_desktop", lambda *a: True)
    res = N.send_notification(cfg, "training_failed", "boom")
    assert "webhook" not in res and "email" not in res


def test_webhook_empty_url_is_false():
    assert N.send_webhook("", N.Notification("e", "m", 1.0, "info")) is False


def test_webhook_success(monkeypatch):
    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(N, "urlopen", lambda *a, **k: R())
    assert N.send_webhook("http://x", N.Notification("e", "m", 1.0, "info")) is True


def test_webhook_error_status(monkeypatch):
    class R:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(N, "urlopen", lambda *a, **k: R())
    assert N.send_webhook("http://x", N.Notification("e", "m", 1.0, "info")) is False


def test_webhook_network_error(monkeypatch):
    def boom(*a, **k):
        raise N.URLError("down")
    monkeypatch.setattr(N, "urlopen", boom)
    assert N.send_webhook("http://x", N.Notification("e", "m", 1.0, "info")) is False


def test_email_failure_is_reported_false(monkeypatch):
    # No SMTP server here, so it must report False rather than raise.
    assert N.send_emailsmtp("nobody@localhost", "s", "b",
                            smtp_host="127.0.0.1", smtp_port=1) is False


# --- CLI -------------------------------------------------------------------

def test_notify_requires_message(cfg, capsys):
    assert N.main(cfg, ["cli", "notify"]) == 2
    assert "requires --message" in capsys.readouterr().out


def test_notify_rejects_bad_data_json(cfg, capsys):
    # Regression: invalid --data was silently dropped.
    rc = N.main(cfg, ["cli", "notify", "--message=m", "--data={not json"])
    assert rc == 2
    assert "--data must be valid JSON" in capsys.readouterr().out


def test_notify_accepts_valid_data(cfg):
    assert N.main(cfg, ["cli", "notify", "--message=m",
                        '--data={"k": 1}']) == 0


@pytest.mark.parametrize("argv", [
    ["cli", "notify-history", "--limit=abc"],
    ["cli", "notify-history", "--limit=0"],
    ["cli", "notify-history", "--limit=-5"],
])
def test_notify_history_rejects_bad_limit(cfg, argv, capsys):
    # Each of these raised an uncaught ValueError before.
    assert N.main(cfg, argv) == 2
    assert "[error]" in capsys.readouterr().out


def test_notify_history_empty(cfg, capsys):
    assert N.main(cfg, ["cli", "notify-history"]) == 0
    assert "no notifications" in capsys.readouterr().out


def test_notify_history_prints_entries(cfg, capsys):
    path = _log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_entry(event="training_failed", message="it broke", ts=1000.0) + "\n")
    assert N.main(cfg, ["cli", "notify-history"]) == 0
    out = capsys.readouterr().out
    assert "training_failed" in out and "it broke" in out


def test_notify_help_lists_events(capsys):
    cfg = Config(raw={"paths": {"analysis_dir": "/tmp"}})
    assert N.main(cfg, ["cli", "notify-help"]) == 0
    assert "training_failed" in capsys.readouterr().out
