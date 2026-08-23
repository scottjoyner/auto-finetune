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
    assert state["schema_version"] == 3
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


def test_promoted_plan_source_identities_reconcile_a_missing_name_watermark(tmp_path):
    """A promoted immutable plan remains authoritative if its name watermark is lost."""
    opencode, hermes = _databases(tmp_path, 14, 470)
    cfg = _cfg(tmp_path, opencode, hermes)
    plan = plan_harvest(cfg, min_new_sessions=1, max_batch_hours=100)

    record_harvest(cfg, plan.sources, plan_id=plan.plan_id)
    state_path = tmp_path / "analysis" / "harvest-state.json"
    state = json.loads(state_path.read_text())
    # Reproduce the real mismatch: promoted plan/source identities survived,
    # while the newer source-name watermarks were absent.
    state["sources"] = {}
    state_path.write_text(json.dumps(state))

    reconciled = plan_harvest(cfg, min_new_sessions=1, max_batch_hours=100)
    assert reconciled.total_new == 0
    assert not reconciled.should_harvest
    assert not reconciled.should_train


def test_promotion_and_source_watermarks_are_one_atomic_state_write(tmp_path, monkeypatch):
    opencode, hermes = _databases(tmp_path)
    cfg = _cfg(tmp_path, opencode, hermes)
    plan = plan_harvest(cfg, min_new_sessions=1)
    writes = []

    monkeypatch.setattr("src.harvest.atomic_write_json",
                        lambda path, payload: writes.append((path, payload)))
    record_harvest(cfg, plan.sources, plan_id=plan.plan_id)

    assert len(writes) == 1
    payload = writes[0][1]
    assert payload["promoted_plans"][plan.plan_id]["sources"] == {
        source.source_id: source.total_sessions for source in plan.sources
    }
    assert all(payload["sources"][source.name]["source_id"] == source.source_id
               for source in plan.sources)


def test_runtime_limit_splits_sources_and_defers_oversized_source(tmp_path):
    # At the test config's estimate, 300 sessions are ~5.4h and 500 are ~9.0h.
    opencode, hermes = _databases(tmp_path, 300, 500)
    cfg = _cfg(tmp_path, opencode, hermes)
    cfg.raw["scheduler"] = {"max_batch_hours": 8}
    plan = plan_harvest(cfg, min_new_sessions=1)

    assert plan.should_harvest and plan.should_train
    assert plan.batch_labels == ["opencode"]
    assert plan.estimated_train_hours <= 8
    assert "deferred hermes" in plan.reason


def test_runtime_limit_defers_without_advancing_when_nothing_fits(tmp_path):
    # Harvesting is decoupled from the training budget: both labels are queued
    # for cheap CPU extraction, but NEITHER fits the 8h train batch, so no
    # training promotion happens and planning alone writes no watermark state.
    opencode, hermes = _databases(tmp_path, 500, 500)
    cfg = _cfg(tmp_path, opencode, hermes)
    plan = plan_harvest(cfg, min_new_sessions=1, max_batch_hours=8)

    assert plan.should_harvest
    assert set(plan.harvest_labels) == {"opencode", "hermes"}
    assert not plan.should_train
    assert plan.batch_labels == []
    assert plan.total_new == 1000
    assert "deferred opencode training" in plan.reason
    assert "deferred hermes training" in plan.reason
    assert not (tmp_path / "analysis" / "harvest-state.json").exists()
