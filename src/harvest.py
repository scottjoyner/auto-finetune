"""Fail-closed drift detection and batch planning for trace harvesting."""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Iterable

from src.config import Config
from src.locking import atomic_write_json


@dataclass
class SourceStats:
    name: str
    db_path: str
    total_sessions: int
    last_modified: float
    db_size_bytes: int
    new_sessions: int
    last_harvest: float
    days_since_harvest: float
    error: str | None = None


@dataclass
class HarvestPlan:
    should_harvest: bool
    should_train: bool
    sources: list[SourceStats]
    total_new: int
    estimated_train_hours: float
    batch_labels: list[str]
    reason: str


def _file_stats(path: str) -> tuple[float, int]:
    candidates = [path, f"{path}-wal"]
    mtimes = [os.path.getmtime(p) for p in candidates if os.path.exists(p)]
    return (max(mtimes) if mtimes else 0, os.path.getsize(path) if os.path.exists(path) else 0)


def _get_opencode_stats(db_path: str) -> dict:
    """Read OpenCode's singular ``session`` table through corruption tolerance."""
    if not os.path.exists(db_path):
        return {"total": 0, "last_modified": 0, "size": 0,
                "error": f"configured OpenCode DB missing: {db_path}"}
    db = None
    try:
        from src.db import CorruptDB
        db = CorruptDB(db_path)
        if not db.table_exists("session"):
            raise RuntimeError("OpenCode table 'session' is missing")
        total = db.count("session")
        modified, size = _file_stats(db_path)
        return {"total": total, "last_modified": modified, "size": size}
    except Exception as exc:
        modified, size = _file_stats(db_path)
        return {"total": 0, "last_modified": modified, "size": size, "error": str(exc)}
    finally:
        if db is not None:
            db.close()


def _get_hermes_stats(db_path: str) -> dict:
    """Read Hermes's ``sessions`` table using its actual timestamp columns."""
    if not os.path.exists(db_path):
        return {"total": 0, "last_modified": 0, "size": 0,
                "error": f"configured Hermes DB missing: {db_path}"}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
        try:
            total, last_ts = con.execute(
                "SELECT COUNT(*), MAX(COALESCE(ended_at, started_at)) FROM sessions"
            ).fetchone()
        finally:
            con.close()
        modified, size = _file_stats(db_path)
        # Timestamp shape varies between older/newer Hermes stores; filesystem
        # mtime is only diagnostic, while the session count drives idempotence.
        return {"total": int(total), "last_modified": modified or float(last_ts or 0),
                "size": size}
    except Exception as exc:
        modified, size = _file_stats(db_path)
        return {"total": 0, "last_modified": modified, "size": size, "error": str(exc)}


def _load_harvest_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {"schema_version": 1, "sources": {}}
    import json
    with open(state_path) as handle:
        raw = json.load(handle)
    # Upgrade the original flat {source: {...}} shape in memory.
    if "sources" not in raw:
        raw = {"schema_version": 1,
               "sources": {k: v for k, v in raw.items() if isinstance(v, dict)}}
    return raw


def _source_stat(name: str, path: str, snapshot: dict, state: dict, now: float) -> SourceStats:
    saved = state.get("sources", {}).get(name, {})
    last_harvest = float(saved.get("last_harvest", 0) or 0)
    baseline = int(saved.get("total_at_harvest", 0) or 0)
    total = int(snapshot.get("total", 0) or 0)
    return SourceStats(
        name=name,
        db_path=path,
        total_sessions=total,
        last_modified=float(snapshot.get("last_modified", 0) or 0),
        db_size_bytes=int(snapshot.get("size", 0) or 0),
        new_sessions=max(0, total - baseline),
        last_harvest=last_harvest,
        days_since_harvest=(now - last_harvest) / 86400 if last_harvest else 999,
        error=snapshot.get("error"),
    )


def get_source_stats(cfg: Config) -> list[SourceStats]:
    state_path = os.path.join(cfg.path("analysis_dir"), "harvest-state.json")
    state = _load_harvest_state(state_path)
    now = time.time()
    sources: list[SourceStats] = []

    opencode_db = cfg.get("sources", "opencode", "db_path", default="")
    if opencode_db:
        sources.append(_source_stat("opencode", opencode_db,
                                    _get_opencode_stats(opencode_db), state, now))

    hermes_db = cfg.get("sources", "hermes", "state_db", default="")
    if hermes_db and cfg.get("sources", "hermes", "enabled", default=True):
        sources.append(_source_stat("hermes", hermes_db,
                                    _get_hermes_stats(hermes_db), state, now))
    return sources


def plan_harvest(cfg: Config, min_new_sessions: int = 50,
                 min_days: float = 1.0, max_batch_hours: float = 8.0) -> HarvestPlan:
    if min_new_sessions <= 0:
        raise ValueError("min_new_sessions must be positive")
    sources = get_source_stats(cfg)
    errors = [f"{s.name}: {s.error}" for s in sources if s.error]
    if errors:
        return HarvestPlan(False, False, sources, 0, 0, [],
                           "source inspection failed (fail closed): " + "; ".join(errors))

    total_new = sum(s.new_sessions for s in sources)
    aggregate_trigger = total_new >= min_new_sessions
    targets = [s.name for s in sources if s.new_sessions > 0 and
               (aggregate_trigger or s.new_sessions >= min_new_sessions or
                s.days_since_harvest >= min_days)]
    should_harvest = bool(targets)
    should_train = aggregate_trigger and bool(targets)
    reasons = [f"{s.name}: {s.new_sessions} new sessions" for s in sources if s.name in targets]
    if aggregate_trigger:
        reasons.append(f"{total_new} total new sessions >= {min_new_sessions}")

    epochs = float(cfg.get("train", "num_train_epochs", default=2) or 2)
    est_hours = total_new * 0.01 * 50 * epochs * 65 / 3600
    if est_hours > max_batch_hours:
        reasons.append(f"estimated {est_hours:.1f}h > {max_batch_hours}h limit")
    if not reasons:
        reasons.append("no new data reached the count/time trigger")
    return HarvestPlan(should_harvest, should_train, sources, total_new,
                       est_hours, targets, "; ".join(reasons))


def record_harvest(cfg: Config, sources: Iterable[SourceStats]) -> None:
    """Atomically advance every successfully extracted source baseline once."""
    state_path = os.path.join(cfg.path("analysis_dir"), "harvest-state.json")
    state = _load_harvest_state(state_path)
    state["schema_version"] = 2
    state.setdefault("sources", {})
    now = time.time()
    for source in sources:
        state["sources"][source.name] = {
            "last_harvest": now,
            "total_at_harvest": source.total_sessions,
            "sessions_harvested": source.new_sessions,
        }
    atomic_write_json(state_path, state)


def main(cfg: Config) -> int:
    plan = plan_harvest(cfg)
    print("[harvest-status]")
    for source in plan.sources:
        suffix = f", ERROR={source.error}" if source.error else ""
        print(f"  {source.name}: {source.total_sessions} sessions, "
              f"{source.new_sessions} new, {source.days_since_harvest:.1f} days{suffix}")
    print(f"\n[harvest-plan] should_harvest={plan.should_harvest} "
          f"should_train={plan.should_train}")
    print(f"  total_new={plan.total_new}, est={plan.estimated_train_hours:.1f}h")
    print(f"  batch={plan.batch_labels}")
    print(f"  reason: {plan.reason}")
    return 0 if not any(source.error for source in plan.sources) else 1
