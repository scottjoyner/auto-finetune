"""Offline tests for the deterministic filesystem-only surface of
``src.pipeline_health``. No network, GPU, DB or model downloads are used.
"""
from __future__ import annotations

import json
import os

import pytest

from src import pipeline_health as ph
from src.config import Config


@pytest.fixture
def dirs(tmp_path):
    d = {
        "raw_dir": tmp_path / "raw",
        "cleaned_dir": tmp_path / "cleaned",
        "dataset_dir": tmp_path / "dataset",
    }
    for p in d.values():
        p.mkdir()
    return d


@pytest.fixture
def cfg(dirs):
    return Config(raw={"paths": {key: str(path) for key, path in dirs.items()}})


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _raise_oserror(*_args, **_kwargs):
    raise OSError("walk denied")


# --- check_raw_dir ---------------------------------------------------------

def test_check_raw_dir_missing(tmp_path):
    missing = tmp_path / "absent"
    result = ph.check_raw_dir(
        Config(raw={"paths": {"raw_dir": str(missing)}})
    )
    assert result == {
        "status": "missing",
        "path": str(missing),
        "files": 0,
        "errors": [f"raw_dir does not exist: {missing}"],
    }


def test_check_raw_dir_ok_counts_nested_json_only(dirs, cfg):
    (dirs["raw_dir"] / "sub").mkdir()
    _write_json(dirs["raw_dir"] / "a.json", {"id": 1})
    _write_json(dirs["raw_dir"] / "sub" / "b.json", {"id": 2})
    (dirs["raw_dir"] / "notes.txt").write_text("not json")

    result = ph.check_raw_dir(cfg)
    assert result["status"] == "ok"
    assert result["files"] == 2
    assert result["errors"] == []


def test_check_raw_dir_corrupt_json_does_not_exceed_validation_cap(dirs, cfg):
    # 101 corrupt files: only the first 100 are opened, so the error count
    # proves the documented cap is honoured.
    for i in range(101):
        (dirs["raw_dir"] / f"bad{i:03d}.json").write_text("{not json")

    result = ph.check_raw_dir(cfg)
    assert result["status"] == "degraded"
    assert result["files"] == 101
    assert result["errors"] == ["100 corrupt JSON files in raw_dir"]


def test_check_raw_dir_walk_oserror_is_reported(cfg):
    from unittest.mock import patch

    with patch.object(os, "walk", side_effect=_raise_oserror):
        result = ph.check_raw_dir(cfg)
    assert result["status"] == "error"
    assert result["files"] == 0
    assert result["errors"] == ["walk denied"]


# --- check_cleaned_dir -----------------------------------------------------

def test_check_cleaned_dir_missing(tmp_path):
    missing = tmp_path / "absent"
    result = ph.check_cleaned_dir(
        Config(raw={"paths": {"cleaned_dir": str(missing)}})
    )
    assert result["status"] == "missing"
    assert result["errors"] == [f"cleaned_dir does not exist: {missing}"]


def test_check_cleaned_dir_ok_with_messages_record(dirs, cfg):
    _write_json(dirs["cleaned_dir"] / "ok.json", {"messages": [{"role": "user"}]})
    (dirs["cleaned_dir"] / "README.md").write_text("ignored")

    result = ph.check_cleaned_dir(cfg)
    assert result["status"] == "ok"
    assert result["files"] == 1
    assert result["errors"] == []


def test_check_cleaned_dir_rejects_bad_shape_and_corrupt_json(dirs, cfg):
    _write_json(dirs["cleaned_dir"] / "list.json", [])
    _write_json(dirs["cleaned_dir"] / "no_messages.json", {"payload": 1})
    (dirs["cleaned_dir"] / "broken.json").write_text("{oops")

    result = ph.check_cleaned_dir(cfg)
    assert result["status"] == "degraded"
    assert result["files"] == 3
    assert result["errors"] == ["3 invalid cleaned files"]


def test_check_cleaned_dir_walk_oserror_is_reported(cfg):
    from unittest.mock import patch

    with patch.object(os, "walk", side_effect=_raise_oserror):
        result = ph.check_cleaned_dir(cfg)
    assert result["status"] == "error"
    assert result["errors"] == ["walk denied"]


# --- check_dataset_dir -----------------------------------------------------

def test_check_dataset_dir_missing(tmp_path):
    missing = tmp_path / "absent"
    result = ph.check_dataset_dir(
        Config(raw={"paths": {"dataset_dir": str(missing)}})
    )
    assert result["status"] == "missing"
    assert result["errors"] == [f"dataset_dir does not exist: {missing}"]


def test_check_dataset_dir_ok_counts_nonblank_jsonl_rows(dirs, cfg):
    (dirs["dataset_dir"] / "a.jsonl").write_text(
        '{"id": 1}\n\n{"id": 2}\n   \n'
    )
    (dirs["dataset_dir"] / "b.jsonl").write_text('{"id": 3}\n')
    (dirs["dataset_dir"] / "ignore.txt").write_text("{" )

    result = ph.check_dataset_dir(cfg)
    assert result["status"] == "ok"
    assert result["files"] == 2
    assert result["total_rows"] == 3
    assert result["errors"] == []


def test_check_dataset_dir_corrupt_line_is_degraded_with_filename(dirs, cfg):
    (dirs["dataset_dir"] / "bad.jsonl").write_text('{"ok": 1}\nnot-json\n')

    result = ph.check_dataset_dir(cfg)
    assert result["status"] == "degraded"
    assert result["files"] == 1
    assert result["total_rows"] == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("bad.jsonl:")


def test_check_dataset_dir_listdir_oserror_is_reported(cfg):
    from unittest.mock import patch

    def _boom(*_args, **_kwargs):
        raise OSError("list denied")

    with patch.object(os, "listdir", side_effect=_boom):
        result = ph.check_dataset_dir(cfg)
    assert result["status"] == "error"
    assert result["errors"] == ["list denied"]


# --- check_pipeline --------------------------------------------------------

def _make_healthy(dirs):
    _write_json(dirs["raw_dir"] / "x.json", {"id": 1})
    _write_json(dirs["cleaned_dir"] / "x.json", {"messages": []})
    (dirs["dataset_dir"] / "x.jsonl").write_text('{"messages": []}\n')


def test_check_pipeline_healthy(dirs, cfg):
    _make_healthy(dirs)
    result = ph.check_pipeline(cfg)
    assert result["status"] == "healthy"
    assert result["errors"] == []
    assert result["summary"] == (
        "Pipeline healthy: raw=1 files, cleaned=1 files, dataset=1 files"
    )
    assert set(result["stages"]) == {"raw", "cleaned", "dataset"}


def test_check_pipeline_one_error_is_degraded(dirs, cfg):
    _write_json(dirs["cleaned_dir"] / "x.json", {"messages": []})
    (dirs["dataset_dir"] / "x.jsonl").write_text('{"m": 1}\n')
    # Point raw_dir at a non-existent path to produce exactly one failure.
    cfg.raw["paths"]["raw_dir"] = str(dirs["raw_dir"].parent / "gone")

    result = ph.check_pipeline(cfg)
    assert result["status"] == "degraded"
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("raw: missing - raw_dir does not exist:")
    assert result["summary"].startswith("Pipeline degraded:")


def test_check_pipeline_three_errors_are_broken(tmp_path):
    cfg = Config(raw={
        "paths": {
            "raw_dir": str(tmp_path / "no-raw"),
            "cleaned_dir": str(tmp_path / "no-cleaned"),
            "dataset_dir": str(tmp_path / "no-dataset"),
        }
    })
    result = ph.check_pipeline(cfg)
    assert result["status"] == "broken"
    assert len(result["errors"]) == 3
    assert [e.split(":", 1)[0] for e in result["errors"]] == [
        "raw", "cleaned", "dataset"
    ]
    assert result["summary"].startswith("Pipeline broken:")


# --- validate_synergy ------------------------------------------------------

def test_validate_synergy_no_directories(tmp_path):
    cfg = Config(raw={
        "paths": {
            "raw_dir": str(tmp_path / "no-raw"),
            "cleaned_dir": str(tmp_path / "no-cleaned"),
            "dataset_dir": str(tmp_path / "no-dataset"),
        }
    })
    assert ph.validate_synergy(cfg) == []


def test_validate_synergy_matching_entries_and_derived_subdir(dirs, cfg):
    (dirs["raw_dir"] / "hermes").mkdir()
    _write_json(dirs["raw_dir"] / "flat.json", {"id": 1})
    (dirs["cleaned_dir"] / "hermes").mkdir()
    (dirs["cleaned_dir"] / "hermes-reasoning").mkdir()
    _write_json(dirs["cleaned_dir"] / "flat.json", {"messages": []})
    (dirs["dataset_dir"] / "d.jsonl").write_text('{"m": 1}\n')

    assert ph.validate_synergy(cfg) == []


def test_validate_synergy_orphan_cleaned_entry_is_reported(dirs, cfg):
    (dirs["cleaned_dir"] / "orphan").mkdir()
    (dirs["cleaned_dir"] / "hermes-reasoning").mkdir()
    (dirs["dataset_dir"] / "d.jsonl").write_text('{"m": 1}\n')

    issues = ph.validate_synergy(cfg)
    assert len(issues) == 1
    assert issues[0].startswith("cleaned_dir has entries not found in raw_dir:")
    assert "orphan" in issues[0]
    assert "hermes-reasoning" not in issues[0]


def test_validate_synergy_flat_json_orphan_alone_is_suppressed(dirs, cfg):
    _write_json(dirs["cleaned_dir"] / "flat.json", {"messages": []})
    (dirs["dataset_dir"] / "d.jsonl").write_text('{"m": 1}\n')

    assert ph.validate_synergy(cfg) == []


def test_validate_synergy_separate_orphan_is_still_reported(dirs, cfg):
    _write_json(dirs["raw_dir"] / "flat.json", {"id": 1})
    _write_json(dirs["cleaned_dir"] / "flat.json", {"messages": []})
    (dirs["cleaned_dir"] / "unknown").mkdir()
    (dirs["dataset_dir"] / "d.jsonl").write_text('{"m": 1}\n')

    issues = ph.validate_synergy(cfg)
    assert len(issues) == 1
    assert "unknown" in issues[0]


def test_validate_synergy_cleaned_without_jsonl_is_reported(dirs, cfg):
    _write_json(dirs["cleaned_dir"] / "flat.json", {"messages": []})

    issues = ph.validate_synergy(cfg)
    assert issues == ["cleaned data exists but no .jsonl dataset files found"]

    (dirs["dataset_dir"] / "d.jsonl").write_text('{"m": 1}\n')
    assert ph.validate_synergy(cfg) == []


def test_validate_synergy_cleaned_subdir_counts_as_cleaned_data(dirs, cfg):
    (dirs["raw_dir"] / "hermes").mkdir()
    (dirs["cleaned_dir"] / "hermes").mkdir()

    assert ph.validate_synergy(cfg) == [
        "cleaned data exists but no .jsonl dataset files found"
    ]


# --- main ------------------------------------------------------------------

def _patch_load(monkeypatch, cfg):
    import src.config

    monkeypatch.setattr(src.config, "load", lambda *_args, **_kwargs: cfg)


def test_main_json_output_healthy(dirs, cfg, monkeypatch, capsys):
    _make_healthy(dirs)
    _patch_load(monkeypatch, cfg)

    assert ph.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["health"]["status"] == "healthy"
    assert payload["synergy_issues"] == []


def test_main_human_output_unhealthy(tmp_path, monkeypatch, capsys):
    cfg = Config(raw={
        "paths": {
            "raw_dir": str(tmp_path / "no-raw"),
            "cleaned_dir": str(tmp_path / "no-cleaned"),
            "dataset_dir": str(tmp_path / "no-dataset"),
        }
    })
    _patch_load(monkeypatch, cfg)

    assert ph.main([]) == 1
    out = capsys.readouterr().out
    assert "[pipeline-health] Pipeline broken:" in out
    assert out.count("  ISSUE: ") == 3
    assert "[pipeline-health] synergy: OK" in out


def test_main_human_output_reports_synergy_issues(dirs, cfg, monkeypatch, capsys):
    _make_healthy(dirs)
    _patch_load(monkeypatch, cfg)
    (dirs["dataset_dir"] / "x.jsonl").unlink()

    assert ph.main([]) == 0
    out = capsys.readouterr().out
    assert "[pipeline-health] synergy issues:" in out
    assert "- cleaned data exists but no .jsonl dataset files found" in out
