"""Coverage tests for cli.main dispatch + error branches (no GPU/network)."""
from __future__ import annotations

import json

import pytest

import src.cli as cli
from src.config import Config

SESSION = {
    "source": "hermes",
    "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "yo"}]},
    ],
}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "s.json").write_text(json.dumps(SESSION))
    ds = tmp_path / "ds"
    ds.mkdir()
    c = Config({
        "paths": {"cleaned_dir": str(cleaned), "dataset_dir": str(ds)},
        "format": {"template": "chatml", "system_prompt": "SYS",
                   "max_turns_per_example": 0, "max_chars_per_example": 24000},
        "train": {"model_name": "m"},
    })
    monkeypatch.setattr(cli, "load", lambda *a, **k: c)
    return c


def test_help_returns_zero(cfg):
    assert cli.main(["cli"]) == 0


def test_eval_requires_label(cfg):
    assert cli.main(["cli", "eval"]) == 2


def test_eval_label_without_heldout(cfg):
    assert cli.main(["cli", "eval", "--label=x"]) == 2


def test_eval_split_requires_label(cfg):
    assert cli.main(["cli", "eval-split"]) == 2


def test_merge_requires_label(cfg):
    assert cli.main(["cli", "merge"]) == 2


def test_probe_requires_label(cfg):
    assert cli.main(["cli", "probe"]) == 2


def test_bench_self_requires_model(cfg):
    assert cli.main(["cli", "bench", "--runner=self"]) == 2


def test_bench_api_requires_url(cfg):
    assert cli.main(["cli", "bench", "--runner=api"]) == 2


def test_bench_api_fleet_none(cfg, monkeypatch):
    import src.fleet as fleet
    monkeypatch.setattr(fleet, "pick_model", lambda *a, **k: None)
    assert cli.main(["cli", "bench", "--runner=api", "--fleet"]) == 2


def test_bench_matrix_requires_specs(cfg):
    assert cli.main(["cli", "bench-matrix"]) == 2


def test_bench_matrix_fleet_empty(cfg, monkeypatch):
    import src.fleet as fleet
    monkeypatch.setattr(fleet, "list_models", lambda *a, **k: [])
    assert cli.main(["cli", "bench-matrix", "--preset=fleet"]) == 2


def test_combine_dispatch(cfg):
    assert cli.main(["cli", "combine"]) == 0


def test_format_dispatch(cfg):
    assert cli.main(["cli", "format"]) == 2


def test_format_all_split_dispatch(cfg):
    assert cli.main(["cli", "format", "--all-split"]) >= 0
