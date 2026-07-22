"""Behavioral tests for live trace stats, planning and scheduler argv."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.config import Config
from src.harvest import SourceStats, get_source_stats, plan_harvest, record_harvest
from src.scheduler import Scheduler


def _cfg(tmp_path: Path, opencode: Path, hermes: Path) -> Config:
    return Config({
        "sources": {
            "opencode": {"db_path": str(opencode), "extra_dbs": []},
            "hermes": {"state_db": str(hermes), "enabled": True},
        },
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "dataset_dir": str(tmp_path / "datasets"),
            "analysis_dir": str(tmp_path / "analysis"),
            "lock_dir": str(tmp_path / "locks"),
        },
        "train": {
            "num_train_epochs": 2,
            "output_dir": str(tmp_path / "checkpoints" / "toolcall-v5-3b-default"),
        },
    })


def _databases(tmp_path: Path, opencode_rows: int = 2, hermes_rows: int = 2):
    opencode = tmp_path / "opencode.db"
    con = sqlite3.connect(opencode)
    con.execute("CREATE TABLE session (id TEXT PRIMARY KEY, time_created INTEGER, time_updated INTEGER)")
    con.executemany("INSERT INTO session VALUES (?, ?, ?)",
                    [(f"o{i}", i, i) for i in range(opencode_rows)])
    con.commit(); con.close()

    hermes = tmp_path / "state.db"
    con = sqlite3.connect(hermes)
    con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, ended_at REAL)")
    con.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                    [(f"h{i}", float(i), float(i)) for i in range(hermes_rows)])
    con.commit(); con.close()
    return opencode, hermes


def test_source_specific_live_sqlite_schemas(tmp_path):
    opencode, hermes = _databases(tmp_path, 3, 4)
    stats = {s.name: s for s in get_source_stats(_cfg(tmp_path, opencode, hermes))}
    assert stats["opencode"].total_sessions == 3
    assert stats["hermes"].total_sessions == 4
    assert stats["opencode"].error is None
    assert stats["hermes"].error is None


def test_missing_configured_source_fails_closed(tmp_path):
    _, hermes = _databases(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "missing.db", hermes)
    plan = plan_harvest(cfg, min_new_sessions=1)
    assert not plan.should_harvest
    assert not plan.should_train
    assert "fail closed" in plan.reason


def test_aggregate_threshold_includes_both_sources(tmp_path):
    opencode, hermes = _databases(tmp_path, 25, 25)
    plan = plan_harvest(_cfg(tmp_path, opencode, hermes), min_new_sessions=50)
    assert plan.should_harvest and plan.should_train
    assert set(plan.batch_labels) == {"opencode", "hermes"}


def test_record_harvest_uses_each_source_total_atomically(tmp_path):
    opencode, hermes = _databases(tmp_path)
    cfg = _cfg(tmp_path, opencode, hermes)
    sources = get_source_stats(cfg)
    record_harvest(cfg, sources)
    state = json.loads((tmp_path / "analysis" / "harvest-state.json").read_text())
    assert state["schema_version"] == 2
    assert state["sources"]["opencode"]["total_at_harvest"] == 2
    assert state["sources"]["hermes"]["total_at_harvest"] == 2
    assert all(s.new_sessions == 0 for s in get_source_stats(cfg))


def test_scheduler_builds_split_argv_and_distinct_outputs(tmp_path):
    opencode, hermes = _databases(tmp_path)
    scheduler = Scheduler(_cfg(tmp_path, opencode, hermes))
    calls = []

    def fake(cmd, timeout=3600, extra_env=None):
        calls.append((cmd, extra_env or {}))
        return 0, "ok"

    scheduler._run_cmd = fake
    ok, result = scheduler.train(["opencode", "hermes"])
    assert ok, result
    assert [c[0][3:] for c in calls] == [
        ["format", "--label=ssd"],
        ["format", "--source=hermes"],
        ["train", "--label=ssd"],
        ["train", "--source=hermes"],
    ]
    outputs = [c[1]["TRAIN_OUTPUT_DIR"] for c in calls if "TRAIN_OUTPUT_DIR" in c[1]]
    assert outputs[0].endswith("toolcall-v5-3b-ssd")
    assert outputs[1].endswith("toolcall-v5-3b-hermes")
