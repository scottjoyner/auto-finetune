"""Pipeline health evidence from disposable filesystem fixtures only."""
import json
from pathlib import Path

import pytest
from src import pipeline_health as health
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{
        "raw_dir":str(tmp_path/"raw"),
        "cleaned_dir":str(tmp_path/"cleaned"),
        "dataset_dir":str(tmp_path/"dataset"),
    }})


def filled(tmp_path):
    for name in ("raw","cleaned","dataset"):
        (tmp_path/name).mkdir()
    (tmp_path/"raw"/"test.json").write_text(json.dumps({"id":"test"}))
    (tmp_path/"cleaned"/"test.json").write_text(json.dumps({"messages":[]}))
    (tmp_path/"dataset"/"test.jsonl").write_text(json.dumps({"messages":[]})+"\n")
    return cfg(tmp_path)


def test_missing_all_stages_fails_closed(tmp_path):
    report=health.check_pipeline(cfg(tmp_path))
    assert report["status"]=="broken"
    assert len(report["errors"])==3
    assert all(v["status"]=="missing" for v in report["stages"].values())


def test_healthy_pipeline_counts_only_valid_sources(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"raw"/"ignored.txt").write_text("not json")
    (tmp_path/"dataset"/"ignored.txt").write_text("not jsonl")
    report=health.check_pipeline(c)
    assert report["status"]=="healthy"
    assert report["stages"]["raw"]["files"]==1
    assert report["stages"]["cleaned"]["files"]==1
    assert report["stages"]["dataset"]["total_rows"]==1
    assert health.validate_synergy(c)==[]


def test_corrupt_raw_stage_marked_degraded(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"raw"/"bad.json").write_text("{invalid")
    result=health.check_raw_dir(c)
    assert result["status"]=="degraded"
    assert "corrupt JSON" in result["errors"][0]
    assert health.check_pipeline(c)["status"]=="degraded"


def test_cleaned_invalid_shape_and_parse_error(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"cleaned"/"invalid.json").write_text("[]")
    (tmp_path/"cleaned"/"bad.json").write_text("{invalid")
    result=health.check_cleaned_dir(c)
    assert result["files"]==3
    assert result["status"]=="degraded"
    assert "2 invalid cleaned" in result["errors"][0]


def test_dataset_corruption_reports_filename_without_hiding_good_rows(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"dataset"/"broken.jsonl").write_text('{"ok": 1}\n{truncated\n')
    result=health.check_dataset_dir(c)
    assert result["status"]=="degraded"
    assert result["total_rows"]==2
    assert any("broken.jsonl" in err for err in result["errors"])


def test_missing_dataset_yields_synergy_warning(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"dataset"/"test.jsonl").unlink()
    issues=health.validate_synergy(c)
    assert any("no .jsonl" in v for v in issues)


def test_orphaned_cleaned_source_is_detected(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"cleaned"/"orphan-source").mkdir()
    issues=health.validate_synergy(c)
    assert any("orphan-source" in v for v in issues)


def test_allowed_derived_cleaned_dir_not_orphan(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"cleaned"/"hermes-reasoning").mkdir()
    assert health.validate_synergy(c)==[]


def test_main_json_and_human_output_follow_health_status(tmp_path,monkeypatch,capsys):
    c=filled(tmp_path)
    monkeypatch.setattr("src.config.load",lambda:c)
    assert health.main(["--json"])==0
    report=json.loads(capsys.readouterr().out)
    assert report["health"]["status"]=="healthy"
    assert report["synergy_issues"]==[]
    assert health.main([])==0
    assert "synergy: OK" in capsys.readouterr().out
    (tmp_path/"raw"/"test.json").write_text("bad")
    assert health.main([])==1
    assert "ISSUE:" in capsys.readouterr().out


def test_nested_sources_and_files_counted_without_network(tmp_path):
    c=filled(tmp_path)
    (tmp_path/"raw"/"extra").mkdir()
    (tmp_path/"cleaned"/"extra").mkdir()
    (tmp_path/"raw"/"extra"/"n.json").write_text("{}")
    (tmp_path/"cleaned"/"extra"/"n.json").write_text('{"messages":[]}')
    assert health.check_raw_dir(c)["files"]==2
    assert health.check_cleaned_dir(c)["files"]==2
    assert health.validate_synergy(c)==[]
