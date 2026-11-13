"""Synthetic, durable GPU-cost accounting tests using temporary directories."""
from __future__ import annotations

import json
import os

import pytest

from src import cost


class Cfg:
    def __init__(self, tmp):
        self.root = tmp
    def path(self, name):
        assert name == "analysis_dir"
        return str(self.root)
    def get(self, *args, default=None):
        return str(self.root / "checkpoints")


def record(tr, label="alpha", train=1.0, eval_h=0.5, **kwargs):
    return tr.record(label=label, training_hours=train, eval_hours=eval_h, **kwargs)


def test_new_cost_ledger_is_empty(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    assert tr.get_summary() == {"count": 0}
    assert tr.get_history() == []
    assert tr.get_by_label() == {}


def test_record_atomic_persistence_and_reload(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    entry = record(tr, "alpha", 1.5, .25, notes="synthetic", cost_usd=0)
    assert entry.total_hours == pytest.approx(1.75)
    assert entry.entry_id.startswith("run-")
    path = tmp_path / "cost-tracking.json"
    assert path.exists()
    reload = cost.CostTracker(str(tmp_path))
    assert len(reload.entries) == 1
    assert reload.entries[0].notes == "synthetic"


def test_history_returns_most_recent_instead_of_oldest(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    for ts, label in ((100, "old"), (200, "middle"), (300, "latest")):
        record(tr, label, 1.0, timestamp=ts)
    assert [e.label for e in tr.get_history(limit=2)] == ["latest", "middle"]
    assert [e.label for e in tr.get_history(label="old")] == ["old"]


def test_summary_totals_and_labels_are_consistent(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    record(tr, "alpha", 2, .5, cost_usd=1.5)
    record(tr, "alpha", 1, 1, cost_usd=2.5)
    record(tr, "beta", 2, 0, cost_usd=0)
    summary = tr.get_summary()
    assert summary["count"] == 3
    assert summary["training_hours"] == 5
    assert summary["eval_hours"] == 1.5
    assert summary["total_hours"] == 6.5
    assert summary["total_cost_usd"] == 4
    assert sorted(summary["labels"]) == ["alpha", "beta"]
    assert tr.get_summary("alpha")["count"] == 2
    assert tr.get_summary("missing") == {"count": 0}


def test_grouped_cost_and_last_run_timestamp(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    record(tr, "alpha", 1, 0, timestamp=100)
    record(tr, "alpha", 3, 0, timestamp=200)
    record(tr, "beta", 2, 0, timestamp=300)
    groups = tr.get_by_label()
    assert groups["alpha"]["runs"] == 2
    assert groups["alpha"]["total_hours"] == 4
    assert groups["alpha"]["last_run"] == 200
    assert groups["beta"]["total_hours"] == 2


def test_total_hours_can_be_supplied_explicitly(tmp_path):
    tr = cost.CostTracker(str(tmp_path))
    entry = tr.record(label="synthetic", training_hours=3, total_hours=4)
    assert entry.total_hours == 4
    assert tr.get_summary()["total_hours"] == 4


def test_checkpoint_size_only_uses_isolated_tmp_files(tmp_path):
    model = tmp_path / "models"
    subdir = model / "version"
    subdir.mkdir(parents=True)
    (subdir / "a.bin").write_bytes(b"x" * 512)
    (subdir / "b.bin").write_bytes(b"x" * 512)
    assert cost._get_disk_usage(str(model)) == pytest.approx(1024 / (1024 * 1024))


def test_cost_record_cli_writes_into_disposable_analysis_dir(tmp_path, capsys):
    cfg = Cfg(tmp_path)
    out = tmp_path / "checkpoints" / "toolcall-v5-3b-alpha"
    out.mkdir(parents=True)
    (out / "weights.bin").write_bytes(b"x" * 1024)
    assert cost.main(cfg, ["cli", "cost-record", "--label=alpha", "--hours=1.5",
                           "--eval-hours=.5", "--gpu=syntheticGPU",
                           "--notes=synthetic"]) == 0
    tr = cost.CostTracker(str(tmp_path / "costs"))
    assert tr.entries[0].gpu_name == "syntheticGPU"
    assert tr.entries[0].checkpoints_mb == pytest.approx(1024 / 1048576)
    assert tr.entries[0].total_hours == 2
    assert "recorded" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ["cli", "cost-record", "--hours=1"],
    ["cli", "cost-record", "--label=alpha", "--hours=-3"],
    ["cli", "cost-record", "--label=alpha", "--hours=bad"],
    ["cli", "cost-record", "--label=alpha", "--eval-hours=-1"],
    ["cli", "cost-history", "--limit=0"],
    ["cli", "cost-history", "--limit=invalid"],
])
def test_bad_cli_arguments_fail_without_ledger(tmp_path, args):
    cfg = Cfg(tmp_path)
    assert cost.main(cfg, args) == 2
    assert not (tmp_path / "costs" / "cost-tracking.json").exists()


def test_summary_cli_reports_disposable_costs(tmp_path, capsys):
    cfg = Cfg(tmp_path)
    tr = cost.CostTracker(str(tmp_path / "costs"))
    record(tr, "alpha", 2, 0)
    assert cost.main(cfg, ["cli", "cost-summary", "--label=alpha"]) == 0
    output = capsys.readouterr().out
    assert "runs: 1" in output
    assert "alpha" in output


def test_history_cli_empty_and_populated(tmp_path, capsys):
    cfg = Cfg(tmp_path)
    assert cost.main(cfg, ["cli", "cost-history"]) == 0
    assert "no entries" in capsys.readouterr().out
    tr = cost.CostTracker(str(tmp_path / "costs"))
    record(tr, "alpha", 2, 0, notes="synthetic")
    assert cost.main(cfg, ["cli", "cost-history", "--limit=1"]) == 0
    assert "synthetic" in capsys.readouterr().out


def test_unknown_cli_is_nonmutating_help(tmp_path, capsys):
    cfg = Cfg(tmp_path)
    assert cost.main(cfg, ["cli", "unknown"]) == 0
    assert "Commands:" in capsys.readouterr().out
    assert not (tmp_path / "costs" / "cost-tracking.json").exists()
