"""No GPU, no model, no actual training; test subprocess contract only."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import train_buckets as buckets


@pytest.fixture()
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(buckets,"MANIFEST",tmp_path/"manifest.json")
    monkeypatch.setattr(buckets,"REPORT_PATH",tmp_path/"report.json")
    monkeypatch.setattr(buckets,"CHECKOUT_ROOT",tmp_path)
    (tmp_path/"manifest.json").write_text(json.dumps({
        "buckets":{"general":{"count":13},"planning":{"count":12},"empty":{"count":0}},
    }))
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED","1")
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK","1")
    return tmp_path


@pytest.mark.parametrize("live", [False,True])
def test_no_admission_bypass_reaches_child_or_old_checkout(setup,monkeypatch,live):
    children=[]
    def fake_run(cmd,**kwargs):
        children.append((cmd,kwargs))
        assert kwargs["cwd"]==str(setup)
        assert "TRAIN_ALLOW_UNVERIFIED" not in kwargs["env"]
        assert "AF_SKIP_UNMANAGED_CHECK" not in kwargs["env"]
        assert cmd[:4]==[sys.executable,"-m","src.cli","train"]
        assert ("--dry-run" not in cmd)==live
        return Mock(returncode=0,stdout="dry-run OK")
    monkeypatch.setattr(buckets.subprocess,"run",fake_run)
    argv=["train_buckets.py"]
    if live:argv.append("--live")
    monkeypatch.setattr(sys,"argv",argv)
    assert buckets.main()==0
    assert len(children)==2
    report=json.loads((setup/"report.json").read_text())
    assert report["mode"]==("live" if live else "dry-run")
    assert report["passed"]==2
    assert report["failed"]==0
    assert not (setup/"model.safetensors").exists()


def test_a_child_failure_cannot_be_reported_as_success(setup,monkeypatch):
    monkeypatch.setattr(sys,"argv",["train_buckets.py","--group","planning","--live"])
    monkeypatch.setattr(buckets.subprocess,"run",
                        lambda *a,**kw:Mock(returncode=4,stdout="",stderr="gate hold"))
    assert buckets.main()==1
    report=json.loads((setup/"report.json").read_text())
    assert report["failed"]==1 and report["results"][0]["ok"] is False


def test_missing_manifest_is_fail_closed(setup,monkeypatch):
    (setup/"manifest.json").unlink()
    monkeypatch.setattr(sys,"argv",["train_buckets.py","--live"])
    monkeypatch.setattr(buckets.subprocess,"run",lambda *a,**kw:pytest.fail("must not invoke child"))
    assert buckets.main()==1
    assert not (setup/"report.json").exists()


def test_unrecognized_bucket_group_is_rejected(setup,monkeypatch):
    monkeypatch.setattr(sys,"argv",["train_buckets.py","--group","unknown","--live"])
    monkeypatch.setattr(buckets.subprocess,"run",lambda *a,**kw:pytest.fail("should reject"))
    assert buckets.main()==1


def test_timeout_is_failure_without_training(setup,monkeypatch):
    monkeypatch.setattr(sys,"argv",["train_buckets.py","--group","planning","--live"])
    def fake_run(*_a,**_kw):
        raise subprocess.TimeoutExpired(cmd="mock",timeout=300)
    monkeypatch.setattr(buckets.subprocess,"run",fake_run)
    assert buckets.main()==1
    report=json.loads((setup/"report.json").read_text())
    assert report["results"][0]["summary"]=="TIMEOUT"
