"""Automated model promotion pipeline.

Ties together model registry, feedback, and A/B testing to automatically
promote or rollback models based on performance metrics.

Workflow:
  1. New model registered as candidate
  2. A/B test started against current active model
  3. Inference feedback collected for both models
  4. When enough data collected, A/B test completed
  5. Winner promoted, loser retired
  6. If new model performs worse, automatic rollback
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.alerting import AlertManager
from src.feedback_loop import FeedbackCollector
from src.model_registry import ModelRegistry, ModelVersion
from src.locking import atomic_write_json


@dataclass
class PromotionConfig:
    """Configuration for auto-promotion."""
    min_feedback_samples: int = 50  # Minimum feedback before A/B test completion
    min_score_threshold: float = 0.6  # Minimum score to promote
    max_regression: float = 0.1  # Max allowed regression from active model
    auto_rollback: bool = True
    ab_test_duration_hours: float = 24.0


class AutoPromoter:
    """Automates model promotion based on feedback and A/B testing."""

    def __init__(self, cfg: Config, promo_config: PromotionConfig | None = None):
        self.cfg = cfg
        self.config = promo_config or PromotionConfig()
        self.registry = ModelRegistry(cfg)
        self.feedback = FeedbackCollector(cfg)
        self.alerts = AlertManager(cfg)
        self.state_path = os.path.join(cfg.path("analysis_dir"), "auto-promote-state.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"active_ab_test": None, "last_promotion": 0, "last_rollback": 0}

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def register_and_test(self, label: str, checkpoint_path: str,
                          base_model: str, metrics: dict | None = None) -> str | None:
        """Register a new model and start A/B test against active model."""
        # Register new version
        vid = self.registry.register(
            label=label,
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            metrics=metrics or {},
            notes="Auto-registered for A/B testing",
        )
        print(f"[promote] Registered: {vid}")

        # Start A/B test if there's an active model
        active = self.registry.active_version()
        if active and active.version_id != vid:
            test_id = self.registry.start_ab_test(active.version_id, vid)
            if test_id:
                self.state["active_ab_test"] = test_id
                self._save_state()
                print(f"[promote] A/B test started: {test_id} ({active.version_id} vs {vid})")
                self.alerts.fire("info", "auto_promote",
                                f"A/B test started: {active.version_id} vs {vid}")
                return vid
            else:
                print("[promote] HOLD: A/B start unavailable; candidate remains unpromoted")
                return vid
        else:
            # No incumbent baseline is NOT implicit authorization to launch.
            print(f"[promote] HOLD: no active baseline; {vid} remains candidate")
            return vid

    def record_inference(self, model_version: str, session_id: str,
                         input_text: str, output_text: str,
                         score: float, issues: list[str] | None = None) -> None:
        """Record inference feedback and update A/B test metrics."""
        self.feedback.record(
            session_id=session_id,
            model_version=model_version,
            input_text=input_text,
            output_text=output_text,
            score=score,
            issues=issues,
        )

        # Update A/B test if running
        test_id = self.state.get("active_ab_test")
        if test_id:
            ab = self.registry.get_ab_test(test_id)
            if ab and ab.status == "running":
                self.registry.record_ab_metric(test_id, model_version, "score", score)

    def check_promotion(self) -> dict:
        """Check if any A/B test should be completed and promoted."""
        result = {"action": "none", "details": ""}
        test_id = self.state.get("active_ab_test")
        if not test_id:
            result["details"] = "No active A/B test"
            return result

        ab = self.registry.get_ab_test(test_id)
        if not ab or ab.status != "running":
            self.state["active_ab_test"] = None
            self._save_state()
            result["details"] = "A/B test not running"
            return result

        # Balanced, finite and bounded quality receipts are required: a 50/0
        # study must never allow candidate activation or a misleading winner.
        count_a = ab.metrics_a.get("score_count", 0)
        count_b = ab.metrics_b.get("score_count", 0)
        if not all(isinstance(c, int) and c >= 0 for c in (count_a, count_b)):
            return {"action": "hold", "details": "invalid feedback counts"}
        total = count_a + count_b
        per_arm = max(1, (self.config.min_feedback_samples + 1) // 2)
        if total < self.config.min_feedback_samples or min(count_a, count_b) < per_arm:
            result["details"] = f"Insufficient balanced feedback: A={count_a}, B={count_b}"
            return result

        try:
            elapsed_hours = (time.time() - float(ab.start_time)) / 3600
        except (ValueError, TypeError, OverflowError):
            return {"action": "hold", "details": "invalid A/B timestamp"}
        if not math.isfinite(elapsed_hours) or elapsed_hours < 0:
            return {"action": "hold", "details": "invalid A/B timestamp"}
        if elapsed_hours < self.config.ab_test_duration_hours and total < self.config.min_feedback_samples * 2:
            result["details"] = f"Test still running: {elapsed_hours:.1f}h / {self.config.ab_test_duration_hours}h"
            return result

        score_a = ab.metrics_a.get("score", 0)
        score_b = ab.metrics_b.get("score", 0)
        if not all(isinstance(n, (int, float)) and math.isfinite(n)
                   for n in (score_a, score_b)):
            return {"action": "hold", "details": "invalid feedback scores"}
        avg_a, avg_b = score_a / count_a, score_b / count_b
        if not (0 <= avg_a <= 1 and 0 <= avg_b <= 1):
            return {"action": "hold", "details": "feedback score outside [0, 1]"}

        # Registry completion writes only a research result. The registry
        # must never activate or retire any model as an A/B side effect.
        try:
            winner = self.registry.complete_ab_test(test_id, "score")
        except Exception:
            return {"action": "hold", "details": "A/B completion unavailable"}
        if winner is None:
            return {"action": "hold", "details": "A/B completion unavailable"}
        self.state["active_ab_test"] = None
        self.state["last_review"] = time.time()
        self._save_state()

        if winner != ab.version_b:
            return {"action": "hold", "details": "incumbent outperformed candidate"}
        if avg_b < self.config.min_score_threshold:
            return {"action": "hold", "details": "candidate below minimum score"}
        if avg_a - avg_b > self.config.max_regression:
            return {"action": "hold", "details": "candidate exceeds regression budget"}
        # Quality success is still NOT a deployment or credential grant.
        return {"action": "ready_for_review",
                "details": f"Candidate {winner} passed offline scores; manual release decision required"}

    def status(self) -> dict:
        """Get auto-promotion status."""
        return {
            "active_ab_test": self.state.get("active_ab_test"),
            "last_promotion": self.state.get("last_promotion", 0),
            "last_rollback": self.state.get("last_rollback", 0),
            "registry_summary": self.registry.summary(),
            "feedback_summary": self.feedback.summary(),
        }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Automated model promotion")
    parser.add_argument("--register", action="store_true", help="Register new model and start A/B test")
    parser.add_argument("--label", default=None, help="Model label")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--check", action="store_true", help="Check promotion status")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args(argv)

    cfg = load()
    promoter = AutoPromoter(cfg)

    if args.register:
        if not args.label or not args.checkpoint:
            print("[promote] --label and --checkpoint required")
            return 1
        vid = promoter.register_and_test(args.label, args.checkpoint, args.base_model)
        print(f"[promote] Registered: {vid}")
    elif args.check:
        result = promoter.check_promotion()
        print(f"[promote] Action: {result['action']}")
        print(f"[promote] Details: {result['details']}")
        if result["action"] in ("hold", "ready_for_review"):
            return 78
    elif args.status:
        print(json.dumps(promoter.status(), indent=2, default=str))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
