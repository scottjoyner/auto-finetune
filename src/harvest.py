"""Data drift detection and batch planning for auto-harvesting.

Monitors session databases for new data, estimates when enough new sessions
have accumulated to justify a training run, and plans batches.

Usage:
    python -m src.cli harvest-status
    python -m src.cli harvest-plan [--min-new=50]
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


@dataclass
class SourceStats:
    """Stats for a single data source."""
    name: str
    db_path: str
    total_sessions: int
    last_modified: float  # epoch
    db_size_bytes: int
    new_sessions: int  # since last harvest
    last_harvest: float  # epoch of last harvest
    days_since_harvest: float


@dataclass
class HarvestPlan:
    """Plan for the next harvest/training cycle."""
    should_harvest: bool
    should_train: bool
    sources: list[SourceStats]
    total_new: int
    estimated_train_hours: float
    batch_labels: list[str]
    reason: str


def _get_sqlite_stats(db_path: str) -> dict:
    """Get session count and last modified time from a SQLite DB."""
    if not os.path.exists(db_path):
        return {"total": 0, "last_modified": 0, "size": 0}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total = cursor.fetchone()[0]

        # Last modification
        cursor.execute("SELECT MAX(created_at) FROM sessions")
        last_ts = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total": total,
            "last_modified": last_ts,
            "size": os.path.getsize(db_path),
        }
    except Exception as e:
        return {"total": 0, "last_modified": 0, "size": 0, "error": str(e)}


def _load_harvest_state(state_path: str) -> dict:
    """Load the last harvest state."""
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def _save_harvest_state(state_path: str, state: dict):
    """Save harvest state."""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def get_source_stats(cfg: Config) -> list[SourceStats]:
    """Get stats for all configured data sources."""
    state_path = os.path.join(cfg.path("analysis_dir"), "harvest-state.json")
    state = _load_harvest_state(state_path)
    now = time.time()

    sources = []

    # Opencode source
    opencode_db = cfg.get("sources", "opencode", "db_path", default="")
    if opencode_db and os.path.exists(opencode_db):
        stats = _get_sqlite_stats(opencode_db)
        last_harvest = state.get("opencode", {}).get("last_harvest", 0)
        sessions_since = state.get("opencode", {}).get("total_at_harvest", 0)

        sources.append(SourceStats(
            name="opencode",
            db_path=opencode_db,
            total_sessions=stats["total"],
            last_modified=stats["last_modified"],
            db_size_bytes=stats["size"],
            new_sessions=max(0, stats["total"] - sessions_since),
            last_harvest=last_harvest,
            days_since_harvest=(now - last_harvest) / 86400 if last_harvest else 999,
        ))

    # Hermes source
    hermes_db = cfg.get("sources", "hermes", "state_db", default="")
    if hermes_db and os.path.exists(hermes_db):
        stats = _get_sqlite_stats(hermes_db)
        last_harvest = state.get("hermes", {}).get("last_harvest", 0)
        sessions_since = state.get("hermes", {}).get("total_at_harvest", 0)

        sources.append(SourceStats(
            name="hermes",
            db_path=hermes_db,
            total_sessions=stats["total"],
            last_modified=stats["last_modified"],
            db_size_bytes=stats["size"],
            new_sessions=max(0, stats["total"] - sessions_since),
            last_harvest=last_harvest,
            days_since_harvest=(now - last_harvest) / 86400 if last_harvest else 999,
        ))

    return sources


def plan_harvest(
    cfg: Config,
    min_new_sessions: int = 50,
    min_days: float = 1.0,
    max_batch_hours: float = 8.0,
) -> HarvestPlan:
    """Plan the next harvest/training cycle.

    Args:
        cfg: Configuration
        min_new_sessions: Minimum new sessions to trigger training
        min_days: Minimum days since last harvest
        max_batch_hours: Maximum estimated training time per batch

    Returns:
        HarvestPlan with recommendations
    """
    sources = get_source_stats(cfg)
    total_new = sum(s.new_sessions for s in sources)

    # Decision logic
    reasons = []
    should_harvest = False
    should_train = False
    batch_labels = []

    for s in sources:
        if s.new_sessions >= min_new_sessions:
            should_harvest = True
            reasons.append(f"{s.name}: {s.new_sessions} new sessions")
            batch_labels.append(s.name)
        elif s.days_since_harvest >= min_days:
            should_harvest = True
            reasons.append(f"{s.name}: {s.days_since_harvest:.1f} days since harvest")

    if total_new >= min_new_sessions:
        should_train = True
        reasons.append(f"{total_new} total new sessions >= {min_new_sessions}")

    # Estimate training time (rough: ~65 sec/step, ~50 steps/epoch, 2 epochs)
    steps_per_epoch = 50
    epochs = cfg.get("train", "num_train_epochs", default=2)
    sec_per_step = 65
    est_seconds = total_new * 0.01 * steps_per_epoch * epochs * sec_per_step  # rough scaling
    est_hours = est_seconds / 3600

    if est_hours > max_batch_hours:
        reasons.append(f"estimated {est_hours:.1f}h > {max_batch_hours}h limit")

    if not reasons:
        reasons.append("no new data or enough time hasn't passed")

    return HarvestPlan(
        should_harvest=should_harvest,
        should_train=should_train,
        sources=sources,
        total_new=total_new,
        estimated_train_hours=est_hours,
        batch_labels=batch_labels,
        reason="; ".join(reasons),
    )


def record_harvest(cfg: Config, label: str, sessions_harvested: int):
    """Record a harvest completion."""
    state_path = os.path.join(cfg.path("analysis_dir"), "harvest-state.json")
    state = _load_harvest_state(state_path)

    state[label] = {
        "last_harvest": time.time(),
        "total_at_harvest": sessions_harvested,
        "sessions_harvested": sessions_harvested,
    }

    _save_harvest_state(state_path, state)


def main(cfg: Config) -> int:
    """Show harvest status."""
    sources = get_source_stats(cfg)
    plan = plan_harvest(cfg)

    print("[harvest-status]")
    for s in sources:
        print(f"  {s.name}: {s.total_sessions} sessions, "
              f"{s.new_sessions} new since last harvest, "
              f"{s.days_since_harvest:.1f} days ago")

    print(f"\n[harvest-plan] should_harvest={plan.should_harvest} "
          f"should_train={plan.should_train}")
    print(f"  total_new={plan.total_new}, est={plan.estimated_train_hours:.1f}h")
    print(f"  batch={plan.batch_labels}")
    print(f"  reason: {plan.reason}")

    return 0
