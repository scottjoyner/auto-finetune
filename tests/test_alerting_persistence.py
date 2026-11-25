"""Real local alerting storage and CLI contract; no desktop or web requests."""
import json
from pathlib import Path
from types import SimpleNamespace

from src import alerting
from src.config import Config


def config(tmp_path, webhook=""):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path)},
                       "notify":{"webhook_url":webhook}})


def test_alert_persists_then_acknowledges_once_after_restart(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(alerting.urllib.request, "urlopen",
                        lambda req,timeout=5: calls.append((req,timeout)))
    monkeypatch.setattr("subprocess.run",lambda *a,**k: None)
    manager=alerting.AlertManager(config(tmp_path, "https://invalid.test/webhook"))
    aid=manager.fire("error","registry","blocked promotion",{"reason":"no receipt"})
    assert len(calls)==1
    req,timeout=calls[0]
    assert timeout==5
    assert req.get_header("Content-type")=="application/json"
    assert json.loads(req.data)["metadata"]=={"reason":"no receipt"}
    assert manager.summary()["by_level"]["error"]==1
    assert manager.acknowledge(aid) is True
    assert manager.acknowledge("nonexistent") is False
    assert not manager.unacknowledged()
    assert len((tmp_path/"alerts.jsonl").read_text().splitlines())==2

    replay=alerting.AlertManager(config(tmp_path))
    assert len(replay.alerts)==1
    assert replay.alerts[0].acknowledged is True
    assert replay.summary()["total"]==1
    assert replay.recent(level="warning")==[]
    assert replay.recent(level="error",limit=1)[0].message=="blocked promotion"


def test_torn_journal_line_does_not_hide_later_valid_entry(tmp_path):
    alert_path=tmp_path/"alerts.jsonl"
    entries=[
        dict(alert_id="id-1",level="info",source="foo",message="first",timestamp=1.0),
        dict(alert_id="id-2",level="critical",source="bar",message="last",timestamp=2.0),
    ]
    alert_path.write_text(json.dumps(entries[0])+"\n"+"{TRUNCATED\n"+json.dumps(entries[1])+"\n")
    mgr=alerting.AlertManager(config(tmp_path))
    assert [r.message for r in mgr.recent()]==["last","first"]
    assert mgr.summary()["unacknowledged"]==2


def test_notifications_fail_closed_to_local_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(alerting.urllib.request,"urlopen",
                        lambda *a,**k: (_ for _ in ()).throw(OSError("network down")))
    monkeypatch.setattr("subprocess.run",
                        lambda *a,**k: (_ for _ in ()).throw(FileNotFoundError("notify-send")))
    mgr=alerting.AlertManager(config(tmp_path,"https://invalid.test/webhook"))
    assert mgr.fire("critical","watcher","offline")
    assert "offline" in (tmp_path/"alerts.log").read_text()
    assert len(mgr.unacknowledged())==1


def test_cli_summary_and_list_read_only_from_disposable_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(alerting,"load",lambda:config(tmp_path))
    mgr=alerting.AlertManager(config(tmp_path))
    mgr._save_alert(alerting.Alert("local","warning","test","local event",7.0))
    assert alerting.main(["--summary"])==0
    data=json.loads(capsys.readouterr().out)
    assert data["by_level"]["warning"]==1
    assert alerting.main(["--list","--level","warning"])==0
    assert "local event" in capsys.readouterr().out
    assert alerting.main([])==0
    assert "usage:" in capsys.readouterr().out.lower()
