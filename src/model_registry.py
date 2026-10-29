"""Model versioning and A/B testing registry.

Tracks adapter versions, metrics, and enables A/B testing between models.
Stores version metadata in a JSON registry file.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
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
        # Existing malformed state must not silently become an empty registry:
        # that could erase active model identity and A/B review history on save.
        if not os.path.exists(self.registry_path):
            return
        with open(self.registry_path) as stream:
            data = json.load(stream)
        if (type(data) is not dict
                or type(data.get("versions")) is not list
                or type(data.get("ab_tests")) is not list):
            raise ValueError("malformed model registry: expected version and A/B arrays")
        versions = {}
        tests = {}
        for record in data["versions"]:
            if type(record) is not dict:
                raise ValueError("malformed registry model version")
            version = ModelVersion(**record)
            if type(version.version_id) is not str or not version.version_id:
                raise ValueError("malformed model version identity")
            if version.version_id in versions:
                raise ValueError("duplicate persisted model version identity")
            versions[version.version_id] = version
        for record in data["ab_tests"]:
            if type(record) is not dict:
                raise ValueError("malformed persisted A/B comparison")
            ab = ABTest(**record)
            if type(ab.test_id) is not str or not ab.test_id:
                raise ValueError("malformed A/B test identity")
            if ab.test_id in tests:
                raise ValueError("duplicate persisted A/B test identity")
            if (ab.version_a not in versions or ab.version_b not in versions
                    or ab.version_a == ab.version_b):
                raise ValueError("persisted A/B test references invalid models")
            tests[ab.test_id] = ab
        self._versions, self._ab_tests = versions, tests

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
        # Second-resolution IDs silently overwrite earlier registrations.
        # An opaque UUID never derives authority from untrusted model labels.
        version_id = f"model-{uuid.uuid4().hex}"
        if version_id in self._versions:
            raise ValueError("duplicate model registration ID")
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

    def promote(self, version_id: str, *, operator_confirmed: bool = False) -> bool:
        """Activate only after the caller explicitly confirms model promotion."""
        if operator_confirmed is not True:
            return False
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
        if (version_a not in self._versions or version_b not in self._versions
                or version_a == version_b):
            return None
        # A model cannot participate in two simultaneously running A/B tests.
        # Distinct test IDs are required even for tests started in one second.
        if any(t.status == "running" and
               (version_a in (t.version_a, t.version_b)
                or version_b in (t.version_a, t.version_b))
               for t in self._ab_tests.values()):
            return None
        test_id = f"ab-{uuid.uuid4().hex}"
        if test_id in self._ab_tests:
            return None
        ab = ABTest(
            test_id=test_id,
            version_a=version_a,
            version_b=version_b,
            start_time=time.time(),
        )
        self._ab_tests[test_id] = ab
        self._versions[version_a].ab_test_group = test_id
        self._versions[version_b].ab_test_group = test_id
        # Preserve the current active model while the candidate is evaluated.
        # An A/B test does NOT interrupt the serving baseline.
        if self._versions[version_a].status != "active":
            self._versions[version_a].status = "testing"
        if self._versions[version_b].status != "active":
            self._versions[version_b].status = "testing"
        self._save()
        return test_id

    def record_ab_metric(self, test_id: str, version: str, metric: str, value: float) -> bool:
        if (type(metric) is not str or not metric or metric.endswith("_count")
                or type(value) not in (float, int) or not math.isfinite(value)
                or (metric == "score" and not 0 <= value <= 1)):
            return False
        ab = self._ab_tests.get(test_id)
        if not ab or ab.status != "running":
            return False
        if version == ab.version_a:
            metrics = ab.metrics_a
        elif version == ab.version_b:
            metrics = ab.metrics_b
        else:
            return False
        old_total = metrics.get(metric, 0)
        old_count = metrics.get(f"{metric}_count", 0)
        if (type(old_total) not in (int, float) or not math.isfinite(old_total)
                or type(old_count) is not int or not 0 <= old_count < 2**53):
            return False
        new_total = old_total + value
        if not math.isfinite(new_total) or (metric == "score"
                and not 0 <= new_total <= old_count + 1):
            return False
        metrics[metric] = new_total
        metrics[f"{metric}_count"] = old_count + 1
        self._save()
        return True

    def complete_ab_test(self, test_id: str, metric: str = "score",
                         *, operator_confirmed: bool = False) -> str | None:
        """Completing an A/B test also activates its winner; default deny."""
        if operator_confirmed is not True or metric != "score":
            return None
        ab = self._ab_tests.get(test_id)
        if not ab or ab.status != "running":
            return None
        count_a = ab.metrics_a.get("score_count")
        count_b = ab.metrics_b.get("score_count")
        sum_a = ab.metrics_a.get("score")
        sum_b = ab.metrics_b.get("score")
        if (type(count_a) is not int or type(count_b) is not int
                or min(count_a, count_b) < 1
                or type(sum_a) not in (int, float)
                or type(sum_b) not in (int, float)
                or not math.isfinite(sum_a) or not math.isfinite(sum_b)
                or not 0 <= sum_a <= count_a
                or not 0 <= sum_b <= count_b):
            return None
        if ab.version_a not in self._versions or ab.version_b not in self._versions:
            return None
        avg_a, avg_b = sum_a / count_a, sum_b / count_b
        if not math.isfinite(avg_a) or not math.isfinite(avg_b):
            return None
        winner = ab.version_a if avg_a >= avg_b else ab.version_b
        loser = ab.version_b if winner == ab.version_a else ab.version_a
        # Commit one coherent in-memory state and save it once. No intermediate
        # activate-and-rollback side effect, and no mutation on invalid metrics.
        ab.end_time = time.time()
        ab.status = "completed"
        ab.winner = winner
        for model in self._versions.values():
            if model.status == "active" and model.version_id != winner:
                model.status = "retired"
        self._versions[winner].status = "active"
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
    prom.add_argument("--confirm-activation", action="store_true")

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
    abc.add_argument("--confirm-activation", action="store_true")

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
        if registry.promote(args.version, operator_confirmed=args.confirm_activation):
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
        winner = registry.complete_ab_test(args.test, args.metric,
                                           operator_confirmed=args.confirm_activation)
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
