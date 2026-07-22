"""GPU hours and resource usage tracking.

Tracks training time, disk usage, and costs across runs.

Usage:
    python -m src.cli cost-record --label=<name> --hours=8.5
    python -m src.cli cost-summary
    python -m src.cli cost-history
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


COSTS_FILE = "cost-tracking.json"


@dataclass
class CostEntry:
    """A resource usage entry."""
    entry_id: str
    label: str
    timestamp: float
    # Time
    training_hours: float
    eval_hours: float | None = None
    total_hours: float | None = None
    # GPU
    gpu_name: str | None = None
    gpu_hours: float | None = None
    # Disk
    disk_usage_mb: float | None = None
    checkpoints_mb: float | None = None
    # Cost estimate (if applicable)
    cost_usd: float | None = None
    # Metadata
    notes: str | None = None
    run_id: str | None = None


class CostTracker:
    """Track resource usage and costs."""

    def __init__(self, base_dir: str):
        self.costs_path = os.path.join(base_dir, COSTS_FILE)
        self.entries: list[CostEntry] = []
        self._load()

    def _load(self):
        if os.path.exists(self.costs_path):
            with open(self.costs_path) as f:
                data = json.load(f)
            self.entries = [CostEntry(**e) for e in data]

    def _save(self):
        os.makedirs(os.path.dirname(self.costs_path), exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(self.costs_path, "w") as f:
            json.dump(data, f, indent=2)

    def record(self, **kwargs) -> CostEntry:
        """Record a new cost entry."""
        entry_id = f"run-{int(time.time())}"
        kwargs.setdefault("entry_id", entry_id)
        kwargs.setdefault("timestamp", time.time())

        # Auto-compute total_hours
        if kwargs.get("total_hours") is None:
            training = kwargs.get("training_hours", 0) or 0
            eval_h = kwargs.get("eval_hours", 0) or 0
            kwargs["total_hours"] = training + eval_h

        entry = CostEntry(**kwargs)
        self.entries.append(entry)
        self._save()
        return entry

    def get_history(self, label: str | None = None, limit: int = 50) -> list[CostEntry]:
        entries = self.entries
        if label:
            entries = [e for e in entries if e.label == label]
        return sorted(entries, key=lambda e: -e.timestamp)[-limit:]

    def get_summary(self, label: str | None = None) -> dict:
        """Get summary statistics."""
        entries = self.entries
        if label:
            entries = [e for e in entries if e.label == label]

        if not entries:
            return {"count": 0}

        total_hours = sum(e.total_hours or 0 for e in entries)
        training_hours = sum(e.training_hours or 0 for e in entries)
        eval_hours = sum(e.eval_hours or 0 for e in entries)
        total_cost = sum(e.cost_usd or 0 for e in entries)

        return {
            "count": len(entries),
            "total_hours": round(total_hours, 2),
            "training_hours": round(training_hours, 2),
            "eval_hours": round(eval_hours, 2),
            "total_cost_usd": round(total_cost, 2),
            "avg_hours_per_run": round(total_hours / len(entries), 2),
            "labels": list(set(e.label for e in entries)),
        }

    def get_by_label(self) -> dict[str, dict]:
        """Get costs grouped by label."""
        by_label: dict[str, list[CostEntry]] = {}
        for e in self.entries:
            by_label.setdefault(e.label, []).append(e)

        result = {}
        for label, entries in by_label.items():
            total_hours = sum(e.total_hours or 0 for e in entries)
            result[label] = {
                "runs": len(entries),
                "total_hours": round(total_hours, 2),
                "last_run": max(e.timestamp for e in entries),
            }

        return result


def _get_disk_usage(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for cost commands."""
    cmd = argv[1] if len(argv) > 1 else "cost-summary"

    analysis_dir = cfg.path("analysis_dir")
    costs_dir = os.path.join(analysis_dir, "costs")
    tracker = CostTracker(costs_dir)

    label = None
    training_hours = 0.0
    eval_hours = None
    notes = None
    gpu_name = None

    for arg in argv:
        if arg.startswith("--label="):
            label = arg.split("=", 1)[1]
        elif arg.startswith("--hours="):
            training_hours = float(arg.split("=", 1)[1])
        elif arg.startswith("--eval-hours="):
            eval_hours = float(arg.split("=", 1)[1])
        elif arg.startswith("--notes="):
            notes = arg.split("=", 1)[1]
        elif arg.startswith("--gpu="):
            gpu_name = arg.split("=", 1)[1]

    if cmd == "cost-record":
        if not label:
            print("[error] cost-record requires --label=<name>")
            return 2

        # Auto-detect disk usage
        out_base = cfg.get("train", "output_dir",
                          default="/media/scott/data/finetune-staging/outputs/checkpoints")
        checkpoint_dir = os.path.join(out_base, f"toolcall-v5-3b-{label}")
        checkpoints_mb = None
        if os.path.exists(checkpoint_dir):
            checkpoints_mb = _get_disk_usage(checkpoint_dir)

        entry = tracker.record(
            label=label,
            training_hours=training_hours,
            eval_hours=eval_hours,
            gpu_name=gpu_name,
            checkpoints_mb=checkpoints_mb,
            notes=notes,
        )

        print(f"[cost-record] recorded {entry.entry_id}")
        print(f"  label: {label}")
        print(f"  training: {training_hours:.1f}h")
        if eval_hours:
            print(f"  eval: {eval_hours:.1f}h")
        print(f"  total: {entry.total_hours:.1f}h")
        if checkpoints_mb:
            print(f"  checkpoints: {checkpoints_mb:.0f}MB")
        return 0

    if cmd == "cost-summary":
        label_filter = label
        summary = tracker.get_summary(label_filter)
        by_label = tracker.get_by_label()

        print(f"[cost-summary]")
        print(f"  runs: {summary['count']}")
        print(f"  total hours: {summary.get('total_hours', 0):.1f}")
        print(f"  training hours: {summary.get('training_hours', 0):.1f}")
        print(f"  eval hours: {summary.get('eval_hours', 0):.1f}")
        print(f"  avg hours/run: {summary.get('avg_hours_per_run', 0):.1f}")

        if by_label:
            print(f"\n  by label:")
            for lbl, stats in sorted(by_label.items(), key=lambda x: -x[1]['total_hours']):
                print(f"    {lbl}: {stats['runs']} runs, {stats['total_hours']:.1f}h")
        return 0

    if cmd == "cost-history":
        limit = 20
        for arg in argv:
            if arg.startswith("--limit="):
                limit = int(arg.split("=", 1)[1])

        history = tracker.get_history(label, limit)
        if not history:
            print("[cost-history] no entries")
            return 0

        print(f"[cost-history] {len(history)} entries:")
        for e in history:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
            hours = e.total_hours or 0
            print(f"  [{e.label}] {ts} {hours:.1f}h")
            if e.notes:
                print(f"    {e.notes}")
        return 0

    print("Commands:")
    print("  cost-record --label=<name> --hours=<h> [--eval-hours=<h>] [--gpu=<name>]")
    print("  cost-summary [--label=<name>]")
    print("  cost-history [--label=<name>] [--limit=N]")
    return 0
