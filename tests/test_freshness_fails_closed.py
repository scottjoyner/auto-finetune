"""Real temporary-file and synthetic-clock data freshness safety tests."""
from __future__ import annotations

import json
import os

import pytest

from src import data_freshness as f


class Cfg:
    def __init__(self, root):
        self.root = root
    def path(self, name):
        assert name in {"raw_dir", "cleaned_dir", "dataset_dir", "analysis_dir"}
        return str(self.root / name)


def make_paths(tmp_path):
    for x in ("raw_dir", "cleaned_dir", "dataset_dir", "analysis_dir"):
        (tmp_path / x).mkdir()
    return Cfg(tmp_path)


def set_mtime(p, now, age_h):
    p.write_text('{"synthetic":true}\n')
    ts = now - age_h * 3600
    os.utime(p, (ts, ts))


@pytest.mark.parametrize(("age", "status"), [
    (0.1, "fresh"), (1.2, "aging"), (24.1, "stale"),
    (168.1, "critical"),
])
def test_age_bands_and_report_persist(tmp_path, monkeypatch, age, status):
    cfg = make_paths(tmp_path)
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    for stage, ext in (("raw_dir", ".json"), ("cleaned_dir", ".json"),
                       ("dataset_dir", ".jsonl")):
        set_mtime(tmp_path / stage / ("record" + ext), now, age)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: age)
    report = f.assess_freshness(cfg)
    assert report.overall_staleness == status
    saved = json.loads((tmp_path / "analysis_dir" / "freshness-report.json").read_text())
    assert saved["overall_staleness"] == status
    assert saved["timestamp"] == now


def test_all_stages_missing_is_critical(tmp_path, monkeypatch):
    cfg = make_paths(tmp_path)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: float("inf"))
    report = f.assess_freshness(cfg)
    assert report.overall_staleness == "critical"
    assert any("CRITICAL" in s for s in report.recommendations)


@pytest.mark.parametrize("missing", ["raw_dir", "cleaned_dir", "dataset_dir"])
def test_single_missing_stage_cannot_be_masked_by_fresh_files(
        tmp_path, monkeypatch, missing):
    cfg = make_paths(tmp_path)
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    for stage, ext in (("raw_dir", ".json"), ("cleaned_dir", ".json"),
                       ("dataset_dir", ".jsonl")):
        if stage != missing:
            set_mtime(tmp_path / stage / ("record" + ext), now, 0.5)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: 0.5)
    r = f.assess_freshness(cfg)
    assert r.overall_staleness == "critical"


def test_recency_tracks_newest_nested_json_and_ignores_wrong_extension(
        tmp_path, monkeypatch):
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    a = tmp_path / "raw"
    a.mkdir()
    child = a / "nested"
    child.mkdir()
    set_mtime(a / "old.json", now, 5)
    set_mtime(child / "new.json", now, 2)
    set_mtime(child / "ignored.txt", now, 0.1)
    assert f._newest_file_age(str(a)) == pytest.approx(2)


def test_missing_file_after_walk_is_ignored(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    directory.mkdir()
    monkeypatch.setattr(f.os, "walk", lambda p: [(str(directory), [], ["missing.json"])])
    assert f._newest_file_age(str(directory)) == float("inf")


def test_checkpoint_age_safetensors_and_bin(tmp_path, monkeypatch):
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    checkpoint = tmp_path / "models"
    checkpoint.mkdir()
    child = checkpoint / "v1"
    child.mkdir()
    set_mtime(checkpoint / "one.safetensors", now, 8)
    set_mtime(child / "two.bin", now, 3)
    set_mtime(child / "ignore.json", now, 1)
    assert f._newest_model_age(str(checkpoint)) == pytest.approx(3)


def test_missing_or_empty_checkpoint_ages_are_unknown(tmp_path):
    assert f._newest_model_age(str(tmp_path / "missing")) == float("inf")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert f._newest_model_age(str(checkpoint)) == float("inf")


def test_model_file_disappears_during_scan(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(f.os, "walk", lambda p: [(str(checkpoint), [], ["vanished.bin"])])
    assert f._newest_model_age(str(checkpoint)) == float("inf")


def test_stale_stage_recommendations_include_pipeline_followups(tmp_path, monkeypatch):
    cfg = make_paths(tmp_path)
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    for name, ext, age in (("raw_dir", ".json", 49),
                            ("cleaned_dir", ".json", 64),
                            ("dataset_dir", ".jsonl", 90)):
        set_mtime(tmp_path / name / ("r" + ext), now, age)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: 200)
    rec = f.assess_freshness(cfg).recommendations
    assert any("Raw data" in x for x in rec)
    assert any("Cleaned data" in x for x in rec)
    assert any("Dataset lags" in x for x in rec)
    assert any("Model is stale" in x for x in rec)


def test_report_write_error_does_not_produce_false_freshness(tmp_path, monkeypatch):
    cfg = make_paths(tmp_path)
    monkeypatch.setattr(f, "atomic_write_json", lambda *a: (_ for _ in ()).throw(OSError("write error")))
    report = f.assess_freshness(cfg)
    assert report.overall_staleness == "critical"


@pytest.mark.parametrize("json_mode", [True, False])
def test_freshness_cli_uses_synthetic_assessment(
        tmp_path, monkeypatch, capsys, json_mode):
    cfg = make_paths(tmp_path)
    monkeypatch.setattr(f, "load", lambda: cfg)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: float("inf"))
    assert f.main(["--json"] if json_mode else []) == 0
    output = capsys.readouterr().out
    if json_mode:
        assert json.loads(output)["overall_staleness"] == "critical"
    else:
        assert "Overall: critical" in output


def test_fresh_data_cannot_override_missing_durable_receipt(tmp_path, monkeypatch):
    cfg = make_paths(tmp_path)
    now = 1_900_000_000.0
    monkeypatch.setattr(f.time, "time", lambda: now)
    for stage, ext in (("raw_dir", ".json"), ("cleaned_dir", ".json"),
                       ("dataset_dir", ".jsonl")):
        set_mtime(tmp_path / stage / ("record" + ext), now, 0.25)
    monkeypatch.setattr(f, "_newest_model_age", lambda p: 0.25)
    monkeypatch.setattr(f, "atomic_write_json", lambda *a: (_ for _ in ()).throw(OSError("denied")))
    result = f.assess_freshness(cfg)
    assert result.overall_staleness == "critical"
    assert any("receipt unavailable" in x for x in result.recommendations)
