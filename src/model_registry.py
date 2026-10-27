"""Model versioning and A/B testing registry.

Tracks adapter versions, metrics, and enables A/B testing between models.
Stores version metadata in a JSON registry file.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class ModelVersion:
    """A versioned model adapter."""
    version_id: str
    label: str
    checkpoint_path: str
    base_model: str
    created_at: float
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "candidate"  # candidate, active, retired, testing
    parent_version: str | None = None
    notes: str = ""
    ab_test_group: str | None = None


@dataclass
class ABTest:
    """A/B test configuration between two model versions."""
    test_id: str
    version_a: str
    version_b: str
    start_time: float
    end_time: float | None = None
    metrics_a: dict[str, float] = field(default_factory=dict)
    metrics_b: dict[str, float] = field(default_factory=dict)
    winner: str | None = None
    status: str = "running"  # running, completed, cancelled


class ModelRegistry:
    """Manages model versions and A/B tests."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.registry_path = os.path.join(cfg.path("analysis_dir"), "model-registry.json")
        self._versions: dict[str, ModelVersion] = {}
        self._ab_tests: dict[str, ABTest] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                for v in data.get("versions", []):
                    mv = ModelVersion(**v)
                    self._versions[mv.version_id] = mv
                for t in data.get("ab_tests", []):
                    ab = ABTest(**t)
                    self._ab_tests[ab.test_id] = ab
            except Exception:
                pass

    def _save(self) -> None:
        data = {
            "versions": [asdict(v) for v in self._versions.values()],
            "ab_tests": [asdict(t) for t in self._ab_tests.values()],
            "updated_at": time.time(),
        }
        atomic_write_json(self.registry_path, data)

    def register(self, label: str, checkpoint_path: str, base_model: str,
                 metrics: dict | None = None, parent: str | None = None,
                 notes: str = "") -> str:
        """Register a new model version. Returns version_id."""
        version_id = f"{label}-{int(time.time())}"
        mv = ModelVersion(
            version_id=version_id,
            label=label,
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            created_at=time.time(),
            metrics=metrics or {},
            parent_version=parent,
            notes=notes,
        )
        self._versions[version_id] = mv
        self._save()
        return version_id

    def get(self, version_id: str) -> ModelVersion | None:
        return self._versions.get(version_id)

    def list_versions(self, status: str | None = None) -> list[ModelVersion]:
        versions = list(self._versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def promote(self, version_id: str) -> bool:
        """Promote a candidate to active (retires previous active)."""
        mv = self._versions.get(version_id)
        if not mv:
            return False
        # Retire current active
        for v in self._versions.values():
            if v.status == "active" and v.version_id != version_id:
                v.status = "retired"
        mv.status = "active"
        self._save()
        return True

    def retire(self, version_id: str) -> bool:
        mv = self._versions.get(version_id)
        if not mv:
            return False
        mv.status = "retired"
        self._save()
        return True

    def active_version(self) -> ModelVersion | None:
        for v in self._versions.values():
            if v.status == "active":
                return v
        return None

    def update_metrics(self, version_id: str, metrics: dict) -> bool:
        mv = self._versions.get(version_id)
        if not mv:
            return False
        mv.metrics.update(metrics)
        self._save()
        return True

    def start_ab_test(self, version_a: str, version_b: str) -> str | None:
        """Start an A/B test between two versions."""
        if version_a not in self._versions or version_b not in self._versions:
            return None
        test_id = f"ab-{int(time.time())}"
        ab = ABTest(
            test_id=test_id,
            version_a=version_a,
            version_b=version_b,
            start_time=time.time(),
        )
        self._ab_tests[test_id] = ab
        self._versions[version_a].ab_test_group = test_id
        self._versions[version_b].ab_test_group = test_id
        self._versions[version_a].status = "testing"
        self._versions[version_b].status = "testing"
        self._save()
        return test_id

    def record_ab_metric(self, test_id: str, version: str, metric: str, value: float) -> bool:
        ab = self._ab_tests.get(test_id)
        if not ab or ab.status != "running":
            return False
        if version == ab.version_a:
            ab.metrics_a[metric] = ab.metrics_a.get(metric, 0) + value
            ab.metrics_a[f"{metric}_count"] = ab.metrics_a.get(f"{metric}_count", 0) + 1
        elif version == ab.version_b:
            ab.metrics_b[metric] = ab.metrics_b.get(metric, 0) + value
            ab.metrics_b[f"{metric}_count"] = ab.metrics_b.get(f"{metric}_count", 0) + 1
        else:
            return False
        self._save()
        return True

    def complete_ab_test(self, test_id: str, metric: str = "score") -> str | None:
        """Complete an A/B test and determine winner by metric."""
        ab = self._ab_tests.get(test_id)
        if not ab or ab.status != "running":
            return None
        ab.end_time = time.time()
        ab.status = "completed"

        # Compute average metric
        count_a = ab.metrics_a.get(f"{metric}_count", 1)
        count_b = ab.metrics_b.get(f"{metric}_count", 1)
        avg_a = ab.metrics_a.get(metric, 0) / max(1, count_a)
        avg_b = ab.metrics_b.get(metric, 0) / max(1, count_b)

        winner = ab.version_a if avg_a >= avg_b else ab.version_b
        ab.winner = winner

        # Promote winner
        self.promote(winner)
        # Retire loser
        loser = ab.version_b if winner == ab.version_a else ab.version_a
        if loser in self._versions:
            self._versions[loser].status = "retired"

        self._save()
        return winner

    def get_ab_test(self, test_id: str) -> ABTest | None:
        return self._ab_tests.get(test_id)

    def list_ab_tests(self, status: str | None = None) -> list[ABTest]:
        tests = list(self._ab_tests.values())
        if status:
            tests = [t for t in tests if t.status == status]
        return sorted(tests, key=lambda t: t.start_time, reverse=True)

    def summary(self) -> dict:
        return {
            "total_versions": len(self._versions),
            "active": self.active_version().version_id if self.active_version() else None,
            "candidates": len([v for v in self._versions.values() if v.status == "candidate"]),
            "testing": len([v for v in self._versions.values() if v.status == "testing"]),
            "retired": len([v for v in self._versions.values() if v.status == "retired"]),
            "ab_tests_running": len([t for t in self._ab_tests.values() if t.status == "running"]),
            "ab_tests_completed": len([t for t in self._ab_tests.values() if t.status == "completed"]),
        }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Model registry")
    sub = parser.add_subparsers(dest="action")

    reg = sub.add_parser("register", help="Register a new model version")
    reg.add_argument("--label", required=True)
    reg.add_argument("--checkpoint", required=True)
    reg.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    reg.add_argument("--metrics", default="{}")

    sub.add_parser("list", help="List all versions")
    sub.add_parser("active", help="Show active version")
    sub.add_parser("summary", help="Show registry summary")

    prom = sub.add_parser("promote", help="Promote a version to active")
    prom.add_argument("--version", required=True)

    ab = sub.add_parser("ab-test", help="Start A/B test")
    ab.add_argument("--a", required=True)
    ab.add_argument("--b", required=True)

    abm = sub.add_parser("ab-metric", help="Record A/B test metric")
    abm.add_argument("--test", required=True)
    abm.add_argument("--version", required=True)
    abm.add_argument("--metric", required=True)
    abm.add_argument("--value", type=float, required=True)

    abc = sub.add_parser("ab-complete", help="Complete A/B test")
    abc.add_argument("--test", required=True)
    abc.add_argument("--metric", default="score")

    args = parser.parse_args(argv)

    cfg = load()
    registry = ModelRegistry(cfg)

    if args.action == "register":
        metrics = json.loads(args.metrics)
        vid = registry.register(args.label, args.checkpoint, args.base_model, metrics)
        print(f"[registry] Registered: {vid}")
    elif args.action == "list":
        for v in registry.list_versions():
            print(f"  {v.version_id}: {v.status} (created {v.created_at:.0f})")
    elif args.action == "active":
        v = registry.active_version()
        print(f"Active: {v.version_id if v else 'none'}")
    elif args.action == "summary":
        print(json.dumps(registry.summary(), indent=2))
    elif args.action == "promote":
        if registry.promote(args.version):
            print(f"[registry] Promoted: {args.version}")
        else:
            print(f"[registry] Not found: {args.version}")
    elif args.action == "ab-test":
        tid = registry.start_ab_test(args.a, args.b)
        if tid:
            print(f"[registry] A/B test started: {tid}")
        else:
            print("[registry] Failed to start A/B test")
    elif args.action == "ab-metric":
        if registry.record_ab_metric(args.test, args.version, args.metric, args.value):
            print(f"[registry] Metric recorded")
        else:
            print("[registry] Failed to record metric")
    elif args.action == "ab-complete":
        winner = registry.complete_ab_test(args.test, args.metric)
        if winner:
            print(f"[registry] Winner: {winner}")
        else:
            print("[registry] Failed to complete test")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
