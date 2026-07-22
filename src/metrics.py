"""Training metrics tracking and regression detection.

Records metrics for each training run, compares across versions,
and detects performance regressions.

Usage:
    python -m src.cli metrics-record --label=<name> --loss=0.45 --tool-exact=0.82
    python -m src.cli metrics-compare --label=<name>
    python -m src.cli metrics-history
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


METRICS_FILE = "training-metrics.json"


@dataclass
class TrainingMetrics:
    """Metrics for a single training run."""
    model_id: str
    label: str
    version: int
    timestamp: float
    # Training metrics
    train_loss: float | None = None
    train_loss_final: float | None = None
    train_runtime_seconds: float | None = None
    train_samples_per_second: float | None = None
    # Evaluation metrics
    eval_loss: float | None = None
    eval_perplexity: float | None = None
    # Tool-calling metrics
    tool_exact_match: float | None = None
    tool_partial_match: float | None = None
    tool_recall: float | None = None
    # Dataset info
    dataset_size: int | None = None
    dataset_label: str | None = None
    # Hardware
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None


class MetricsTracker:
    """Track and compare training metrics across versions."""

    def __init__(self, base_dir: str):
        self.metrics_path = os.path.join(base_dir, METRICS_FILE)
        self.metrics: list[TrainingMetrics] = []
        self._load()

    def _load(self):
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path) as f:
                data = json.load(f)
            self.metrics = [TrainingMetrics(**m) for m in data]

    def _save(self):
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        data = [asdict(m) for m in self.metrics]
        with open(self.metrics_path, "w") as f:
            json.dump(data, f, indent=2)

    def record(self, **kwargs) -> TrainingMetrics:
        """Record a new metrics entry."""
        entry = TrainingMetrics(**kwargs)
        self.metrics.append(entry)
        self._save()
        return entry

    def get_history(self, label: str | None = None, limit: int = 50) -> list[TrainingMetrics]:
        """Get metrics history, optionally filtered by label."""
        metrics = self.metrics
        if label:
            metrics = [m for m in metrics if m.label == label]
        return sorted(metrics, key=lambda m: -m.timestamp)[-limit:]

    def get_latest(self, label: str) -> TrainingMetrics | None:
        """Get the latest metrics for a label."""
        candidates = [m for m in self.metrics if m.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.timestamp)

    def get_best(self, label: str, metric: str = "eval_loss") -> TrainingMetrics | None:
        """Get the best performing version by a metric."""
        candidates = [m for m in self.metrics if m.label == label]
        if not candidates:
            return None

        # Filter out None values
        valid = [m for m in candidates if getattr(m, metric, None) is not None]
        if not valid:
            return None

        # Lower is better for loss, higher is better for others
        if "loss" in metric:
            return min(valid, key=lambda m: getattr(m, metric))
        else:
            return max(valid, key=lambda m: getattr(m, metric))

    def detect_regression(
        self,
        label: str,
        threshold: float = 0.05,
        metric: str = "eval_loss",
    ) -> tuple[bool, str]:
        """Detect if the latest version regressed compared to the best.

        Returns:
            (is_regression, message)
        """
        latest = self.get_latest(label)
        best = self.get_best(label, metric)

        if not latest or not best:
            return False, "insufficient data"

        if latest.model_id == best.model_id:
            return False, "latest is best"

        latest_val = getattr(latest, metric, None)
        best_val = getattr(best, metric, None)

        if latest_val is None or best_val is None:
            return False, "metric not available"

        if "loss" in metric:
            # Lower is better - regression if latest is higher
            regression = latest_val > best_val * (1 + threshold)
            direction = "higher"
        else:
            # Higher is better - regression if latest is lower
            regression = latest_val < best_val * (1 - threshold)
            direction = "lower"

        if regression:
            return True, (
                f"regression detected: {metric}={latest_val:.4f} "
                f"(best was {best_val:.4f}, {direction} than threshold)"
            )

        return False, f"OK: {metric}={latest_val:.4f} (best={best_val:.4f})"

    def compare_versions(
        self,
        label: str,
        v1: int | None = None,
        v2: int | None = None,
    ) -> dict:
        """Compare two versions of a model."""
        candidates = sorted(
            [m for m in self.metrics if m.label == label],
            key=lambda m: m.version,
        )

        if len(candidates) < 2:
            return {"error": "need at least 2 versions to compare"}

        # Default: compare latest two
        if v1 is None or v2 is None:
            v1_model = candidates[-2]
            v2_model = candidates[-1]
        else:
            v1_model = next((m for m in candidates if m.version == v1), None)
            v2_model = next((m for m in candidates if m.version == v2), None)
            if not v1_model or not v2_model:
                return {"error": "version not found"}

        comparison = {
            "v1": v1_model.model_id,
            "v2": v2_model.model_id,
            "metrics": {},
        }

        for field in ("train_loss", "eval_loss", "tool_exact_match",
                      "eval_perplexity", "train_runtime_seconds"):
            v1_val = getattr(v1_model, field, None)
            v2_val = getattr(v2_model, field, None)
            if v1_val is not None and v2_val is not None:
                delta = v2_val - v1_val
                pct = (delta / abs(v1_val) * 100) if v1_val != 0 else 0
                comparison["metrics"][field] = {
                    "v1": v1_val,
                    "v2": v2_val,
                    "delta": delta,
                    "pct_change": round(pct, 2),
                }

        return comparison

    def summary(self, label: str | None = None) -> dict:
        """Get a summary of metrics."""
        metrics = self.metrics
        if label:
            metrics = [m for m in metrics if m.label == label]

        if not metrics:
            return {"count": 0}

        losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
        tools = [m.tool_exact_match for m in metrics if m.tool_exact_match is not None]

        return {
            "count": len(metrics),
            "labels": list(set(m.label for m in metrics)),
            "avg_eval_loss": sum(losses) / len(losses) if losses else None,
            "best_eval_loss": min(losses) if losses else None,
            "avg_tool_exact": sum(tools) / len(tools) if tools else None,
            "best_tool_exact": max(tools) if tools else None,
        }


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for metrics commands."""
    cmd = argv[1] if len(argv) > 1 else "metrics-history"

    metrics_dir = os.path.join(cfg.path("analysis_dir"), "metrics")
    tracker = MetricsTracker(metrics_dir)

    if cmd == "metrics-record":
        kwargs = {}
        for arg in argv:
            if arg.startswith("--label="):
                kwargs["label"] = arg.split("=", 1)[1]
            elif arg.startswith("--version="):
                kwargs["version"] = int(arg.split("=", 1)[1])
            elif arg.startswith("--loss="):
                kwargs["train_loss"] = float(arg.split("=", 1)[1])
            elif arg.startswith("--eval-loss="):
                kwargs["eval_loss"] = float(arg.split("=", 1)[1])
            elif arg.startswith("--tool-exact="):
                kwargs["tool_exact_match"] = float(arg.split("=", 1)[1])
            elif arg.startswith("--dataset-size="):
                kwargs["dataset_size"] = int(arg.split("=", 1)[1])
            elif arg.startswith("--runtime="):
                kwargs["train_runtime_seconds"] = float(arg.split("=", 1)[1])

        if "label" not in kwargs:
            print("[error] metrics-record requires --label=<name>")
            return 2

        kwargs.setdefault("model_id", f"toolcall-v5-3b-{kwargs['label']}-v{kwargs.get('version', 0)}")
        kwargs.setdefault("version", 0)
        kwargs.setdefault("timestamp", time.time())

        entry = tracker.record(**kwargs)
        print(f"[metrics-record] recorded {entry.model_id}")
        return 0

    if cmd == "metrics-compare":
        label = None
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]

        if not label:
            print("[error] metrics-compare requires --label=<name>")
            return 2

        comparison = tracker.compare_versions(label)
        if "error" in comparison:
            print(f"[metrics-compare] {comparison['error']}")
            return 1

        print(f"[metrics-compare] {comparison['v1']} vs {comparison['v2']}")
        for metric, vals in comparison.get("metrics", {}).items():
            sign = "+" if vals["delta"] > 0 else ""
            print(f"  {metric}: {vals['v1']:.4f} -> {vals['v2']:.4f} "
                  f"({sign}{vals['delta']:.4f}, {sign}{vals['pct_change']}%)")
        return 0

    if cmd == "metrics-regression":
        label = None
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]

        if not label:
            print("[error] metrics-regression requires --label=<name>")
            return 2

        is_reg, msg = tracker.detect_regression(label)
        if is_reg:
            print(f"[metrics-regression] WARNING: {msg}")
            return 1
        else:
            print(f"[metrics-regression] {msg}")
            return 0

    if cmd == "metrics-history":
        label = None
        limit = 20
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]
            elif arg.startswith("--limit="):
                limit = int(arg.split("=", 1)[1])

        history = tracker.get_history(label, limit)
        if not history:
            print("[metrics-history] no metrics recorded")
            return 0

        print(f"[metrics-history] {len(history)} entries:")
        for m in history:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.timestamp))
            loss = f"loss={m.eval_loss:.4f}" if m.eval_loss else ""
            tool = f"tool={m.tool_exact_match:.3f}" if m.tool_exact_match else ""
            print(f"  {m.model_id} [{ts}] {loss} {tool}")
        return 0

    if cmd == "metrics-summary":
        label = None
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]

        summary = tracker.summary(label)
        print(f"[metrics-summary] {summary['count']} entries")
        if summary.get("avg_eval_loss") is not None:
            print(f"  avg eval_loss: {summary['avg_eval_loss']:.4f}")
            print(f"  best eval_loss: {summary['best_eval_loss']:.4f}")
        if summary.get("avg_tool_exact") is not None:
            print(f"  avg tool_exact: {summary['avg_tool_exact']:.3f}")
            print(f"  best tool_exact: {summary['best_tool_exact']:.3f}")
        return 0

    print("Commands:")
    print("  metrics-record --label=<name> [--loss=<f>] [--eval-loss=<f>] [--tool-exact=<f>]")
    print("  metrics-compare --label=<name>")
    print("  metrics-regression --label=<name>")
    print("  metrics-history [--label=<name>] [--limit=N]")
    print("  metrics-summary [--label=<name>]")
    return 0
