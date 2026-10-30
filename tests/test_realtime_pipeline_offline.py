"""Safety checks on the local realtime pipeline without DB or GPU access."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import realtime_pipeline as pipeline
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{
        "analysis_dir":str(tmp_path/"analysis"),
        "raw_dir":str(tmp_path/"raw"),
        "cleaned_dir":str(tmp_path/"cleaned"),
        "dataset_dir":str(tmp_path/"dataset"),
    }})


def make(tmp_path):
    return pipeline.RealtimePipeline(cfg(tmp_path))


def test_inherited_and_extra_training_bypass_flags_never_reach_child(tmp_path,monkeypatch):
    p=make(tmp_path)
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK","1")
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED","1")
    sent=[]
    def fake_run(cmd,**kwargs):
        sent.append(kwargs)
        assert "AF_SKIP_UNMANAGED_CHECK" not in kwargs["env"]
        assert "TRAIN_ALLOW_UNVERIFIED" not in kwargs["env"]
        assert kwargs["env"]["PYTHONPATH"]==p.repo
        return SimpleNamespace(returncode=0,stdout="OK",stderr="")
    monkeypatch.setattr(pipeline.subprocess,"run",fake_run)
    assert p._run_cmd(["synthetic","train"],extra_env={
        "AF_SKIP_UNMANAGED_CHECK":"1","TRAIN_ALLOW_UNVERIFIED":"1",
        "TEST_ONLY_FLAG":"safe"}) == (0,"OK")
    assert sent[0]["env"]["TEST_ONLY_FLAG"]=="safe"


def test_subprocess_timeout_and_os_error_fail_closed(tmp_path,monkeypatch):
    p=make(tmp_path)
    def timeout(*args,**kwargs):
        raise pipeline.subprocess.TimeoutExpired(args[0],1)
    monkeypatch.setattr(pipeline.subprocess,"run",timeout)
    assert p._run_cmd(["synthetic"])[0]==1
    monkeypatch.setattr(pipeline.subprocess,"run",
                        lambda *a,**k: (_ for _ in ()).throw(OSError("no executable")))
    rc,text=p._run_cmd(["synthetic"])
    assert rc==1
    assert "no executable" in text


def test_training_nonzero_result_cannot_create_model_registry_entry(tmp_path,monkeypatch):
    p=make(tmp_path)
    p.state.new_since_train=51
    monkeypatch.setattr(p,"_run_cmd",lambda cmd,**kw:(78,"admission denied"))
    monkeypatch.setattr(pipeline,"ModelRegistry",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("must not register")))
    assert p._train() is False
    assert p.state.runs_failed==1
    assert p.state.total_trained==0
    assert "admission denied" in p.state.last_error


def test_pipeline_counters_use_only_disposable_directories(tmp_path):
    p=make(tmp_path)
    (tmp_path/"raw"/"nested").mkdir(parents=True)
    (tmp_path/"cleaned").mkdir()
    (tmp_path/"dataset").mkdir()
    (tmp_path/"raw"/"nested"/"one.json").write_text("{}")
    (tmp_path/"raw"/"nested"/"other.txt").write_text("not json")
    (tmp_path/"cleaned"/"two.json").write_text("{}")
    (tmp_path/"dataset"/"a.jsonl").write_text("one\n\nthree\n")
    (tmp_path/"dataset"/"other.dat").write_text("ignored")
    assert p._count_raw_files()==1
    assert p._count_cleaned_files()==1
    assert p._count_dataset_rows()==2


def test_cooldown_and_data_threshold_block_training(tmp_path,monkeypatch):
    p=make(tmp_path)
    monkeypatch.setattr(pipeline.time,"time",lambda:10000.0)
    monkeypatch.setattr(pipeline,"assess_freshness",
                        lambda cfg: SimpleNamespace(overall_staleness="fresh", recommendations=[]))
    assert p._should_train() is False
    p.state.new_since_train=60
    p.state.last_train=10000
    assert p._should_train() is False


def test_critical_freshness_blocks_training_without_subprocess(tmp_path,monkeypatch):
    p=make(tmp_path)
    p.state.new_since_train=100
    monkeypatch.setattr(pipeline,"assess_freshness",
                        lambda cfg:SimpleNamespace(overall_staleness="critical",recommendations=[]))
    monkeypatch.setattr(pipeline,"AlertManager",
                        lambda cfg:SimpleNamespace(fire=lambda *a,**k:"test"))
    assert p._should_train() is False


def test_local_pipeline_state_replay_filters_unknown_fields(tmp_path):
    p=make(tmp_path)
    p.state.new_since_train=29
    p._save_state()
    path=tmp_path/"analysis"/"realtime-pipeline-state.json"
    data=json.loads(path.read_text())
    data["future_flag"]="ignore"
    path.write_text(json.dumps(data))
    assert make(tmp_path).state.new_since_train==29


def test_short_circuit_run_once_if_extraction_fails(tmp_path,monkeypatch):
    p=make(tmp_path)
    monkeypatch.setattr(p,"_extract",lambda:False)
    monkeypatch.setattr(p,"_clean",
                        lambda: (_ for _ in ()).throw(AssertionError("no clean")))
    monkeypatch.setattr(p,"_health_check",lambda:{"status":"fixture"})
    outcome=p.run_once()
    assert outcome["extraction"] is False
    assert outcome["training"] is False
    assert outcome["health"]["status"]=="fixture"


def test_missing_neo4j_secret_denies_extraction_without_process(tmp_path,monkeypatch):
    p=make(tmp_path)
    monkeypatch.delenv("NEO4J_TRACES_PASSWORD",raising=False)
    monkeypatch.setattr(p,"_run_cmd",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("must not execute")))
    assert p._extract() is False
    assert p.state.last_error=="Neo4j traces credential unavailable"


def test_external_secret_is_inherited_without_rewritten_value(tmp_path,monkeypatch):
    p=make(tmp_path)
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD","synthetic-test-credential")
    called=[]
    def fake_command(cmd,**kwargs):
        called.append(kwargs)
        return 1,"synthetic extract failed"
    monkeypatch.setattr(p,"_run_cmd",fake_command)
    assert p._extract() is False
    assert p.state.last_error=="synthetic extract failed"
    assert "NEO4J_TRACES_PASSWORD" not in called[0]["extra_env"]
