"""Real filesystem and synthetic fault coverage for pipeline health release evidence.

Uses temporary directories only; never touches NAS, credentials or Neo4j.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import pipeline_health as health


class FakeConfig:
    def __init__(self, root: Path):
        self.root = root

    def path(self, name: str) -> str:
        assert name in {"raw_dir", "cleaned_dir", "dataset_dir"}
        return str(self.root / name)


@pytest.fixture
def layout(tmp_path):
    cfg = FakeConfig(tmp_path)
    for n in ("raw_dir", "cleaned_dir", "dataset_dir"):
        (tmp_path / n).mkdir()
    return cfg, tmp_path


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("raw_dir", health.check_raw_dir),
        ("cleaned_dir", health.check_cleaned_dir),
        ("dataset_dir", health.check_dataset_dir),
    ],
)
def test_each_missing_directory_denied(tmp_path, name, fn):
    cfg = FakeConfig(tmp_path)
    r = fn(cfg)
    assert r["status"] == "missing"
    assert r["files"] == 0
    assert name in " ".join(r["errors"])


def test_valid_nested_raw_json_and_non_json_ignored(layout):
    cfg, root = layout
    sub = root / "raw_dir" / "opencode"
    sub.mkdir()
    (sub / "a.json").write_text('{"messages": []}')
    (sub / "b.json").write_text('["synthetic"]')
    (sub / "ignored.txt").write_text("not json")
    r = health.check_raw_dir(cfg)
    assert r["status"] == "ok"
    assert r["files"] == 2
    assert not r["errors"]


def test_corrupt_raw_file_degrades(layout):
    cfg, root = layout
    (root / "raw_dir" / "bad.json").write_text("{bad")
    r = health.check_raw_dir(cfg)
    assert r["status"] == "degraded"
    assert "corrupt" in " ".join(r["errors"])


def test_raw_filesystem_failure_denied(layout, monkeypatch):
    cfg, _ = layout
    monkeypatch.setattr(health.os, "walk",
                        lambda *a: (_ for _ in ()).throw(OSError("synthetic os denial")))
    r = health.check_raw_dir(cfg)
    assert r["status"] == "error"
    assert r["errors"]


def test_cleaned_requires_messages_object_and_counts_nested(layout):
    cfg, root = layout
    sub = root / "cleaned_dir" / "hermes"
    sub.mkdir()
    (sub / "valid.json").write_text('{"messages":[]}')
    (sub / "missing_messages.json").write_text("{}")
    (sub / "not_object.json").write_text('["not an object"]')
    (sub / "invalid.json").write_text("{bad")
    r = health.check_cleaned_dir(cfg)
    assert r["status"] == "degraded"
    assert r["files"] == 4
    assert "3 invalid" in " ".join(r["errors"])


def test_cleaned_valid_empty_messages_sufficient(layout):
    cfg, root = layout
    (root / "cleaned_dir" / "ok.json").write_text('{"messages": []}')
    r = health.check_cleaned_dir(cfg)
    assert r["status"] == "ok"
    assert r["files"] == 1


def test_cleaned_os_error_fails_closed(layout, monkeypatch):
    cfg, _ = layout
    monkeypatch.setattr(health.os, "walk",
                        lambda *a: (_ for _ in ()).throw(OSError("synthetic denied")))
    assert health.check_cleaned_dir(cfg)["status"] == "error"


def test_dataset_counts_rows_and_degrades_on_invalid_jsonl(layout):
    cfg, root = layout
    ds = root / "dataset_dir"
    (ds / "good.jsonl").write_text('{"x": 1}\n\n{"y": 2}\n')
    (ds / "bad.jsonl").write_text('{"x": 1}\nINVALID\n')
    (ds / "ignore.txt").write_text("anything")
    r = health.check_dataset_dir(cfg)
    assert r["status"] == "degraded"
    assert r["files"] == 2
    assert r["total_rows"] == 3
    assert any("bad.jsonl" in x for x in r["errors"])


def test_dataset_os_error_fails_closed(layout, monkeypatch):
    cfg, root = layout
    original = health.os.listdir

    def listdir(path):
        if str(path).endswith("dataset_dir"):
            raise PermissionError("synthetic denied")
        return original(path)

    monkeypatch.setattr(health.os, "listdir", listdir)
    assert health.check_dataset_dir(cfg)["status"] == "error"


def test_pipeline_healthy_when_all_stages_well_formed(layout):
    cfg, root = layout
    (root / "raw_dir" / "a.json").write_text('{"ok": true}')
    (root / "cleaned_dir" / "b.json").write_text('{"messages":[]}')
    (root / "dataset_dir" / "c.jsonl").write_text('{"messages":[]}\n')
    result = health.check_pipeline(cfg)
    assert result["status"] == "healthy"
    assert all(x["status"] == "ok" for x in result["stages"].values())
    assert "dataset=1 files" in result["summary"]


def test_pipeline_broken_if_all_three_inputs_missing(tmp_path):
    cfg = FakeConfig(tmp_path)
    result = health.check_pipeline(cfg)
    assert result["status"] == "broken"
    assert len(result["errors"]) == 3


def test_pipeline_degraded_if_one_input_bad(layout):
    cfg, root = layout
    (root / "raw_dir" / "corrupt.json").write_text("not json")
    result = health.check_pipeline(cfg)
    assert result["status"] == "degraded"
    assert len(result["errors"]) == 1


def test_synergy_detects_orphan_cleaned_subdir(layout):
    cfg, root = layout
    (root / "raw_dir" / "opencode").mkdir()
    (root / "cleaned_dir" / "not-raw").mkdir()
    issues = health.validate_synergy(cfg)
    assert any("not found in raw_dir" in issue for issue in issues)


def test_synergy_allows_derived_cleaned_reasoning(layout):
    cfg, root = layout
    (root / "raw_dir" / "hermes").mkdir()
    (root / "cleaned_dir" / "hermes").mkdir()
    (root / "cleaned_dir" / "hermes-reasoning").mkdir()
    (root / "dataset_dir" / "train.jsonl").write_text('{"messages": []}\n')
    assert not health.validate_synergy(cfg)


def test_synergy_flat_raw_and_cleaned_accepted(layout):
    cfg, root = layout
    (root / "raw_dir" / "a.json").write_text("{}")
    (root / "cleaned_dir" / "b.json").write_text('{"messages":[]}')
    (root / "dataset_dir" / "c.jsonl").write_text("{}\n")
    assert health.validate_synergy(cfg) == []


def test_synergy_detects_cleaned_without_formatted_dataset(layout):
    cfg, root = layout
    (root / "raw_dir" / "hermes").mkdir()
    (root / "cleaned_dir" / "hermes").mkdir()
    issues = health.validate_synergy(cfg)
    assert any("no .jsonl dataset" in x for x in issues)


def test_health_json_cli_healthy(layout, monkeypatch, capsys):
    cfg, _ = layout
    import src.config as config
    monkeypatch.setattr(config, "load", lambda: cfg)
    assert health.main(["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["health"]["status"] == "healthy"
    assert out["synergy_issues"] == []


def test_health_text_cli_broken_returns_error(tmp_path, monkeypatch, capsys):
    import src.config as config
    monkeypatch.setattr(config, "load", lambda: FakeConfig(tmp_path))
    assert health.main([]) == 1
    output = capsys.readouterr().out
    assert "Pipeline broken" in output
    assert "ISSUE:" in output
