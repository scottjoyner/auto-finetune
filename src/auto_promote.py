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
        # Research LFM3 adapters must never auto-activate on self-reported
        # scores or the absence of an incumbent model.
        if "lfm3" in base_model.casefold():
            print("[promote] HOLD: LFM3 requires independent promotion approval")
            return vid

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
                print("[promote] HOLD: A/B test unavailable; candidate retained")
                return vid
        else:
            # Initial bootstrap is not proof of model quality or authorization.
            # A separate explicit review/activation step is required.
            print("[promote] HOLD: No active model; candidate retained for review")
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

        # An A/B score is a self-reported measurement, not a signed
        # approval. Never finalize or activate a model in this observer.
        # The registry's legacy complete_ab_test() is intentionally not
        # called here: it activates a winner before reviewing thresholds.
        count_a = ab.metrics_a.get("score_count", 0)
        count_b = ab.metrics_b.get("score_count", 0)
        if (type(count_a) is not int or type(count_b) is not int
                or count_a <= 0 or count_b <= 0):
            result["details"] = "Invalid or missing per-arm feedback counts"
            return result
        total = count_a + count_b
        if total < self.config.min_feedback_samples:
            result["details"] = f"Insufficient feedback: {total}/{self.config.min_feedback_samples}"
            return result
        # A single arm must not dominate the gate.
        if min(count_a, count_b) < max(1, self.config.min_feedback_samples // 4):
            result["details"] = "Insufficient feedback for one A/B arm"
            return result

        sum_a = ab.metrics_a.get("score")
        sum_b = ab.metrics_b.get("score")
        if (type(sum_a) not in (int, float) or type(sum_b) not in (int, float)
                or not math.isfinite(sum_a) or not math.isfinite(sum_b)
                or not 0 <= sum_a <= count_a or not 0 <= sum_b <= count_b):
            result["details"] = "Invalid self-reported A/B scores"
            return result
        elapsed_hours = (time.time() - ab.start_time) / 3600
        if not math.isfinite(elapsed_hours) or elapsed_hours < 0:
            result["details"] = "Invalid A/B experiment timing"
            return result
        if (elapsed_hours < self.config.ab_test_duration_hours
                and total < self.config.min_feedback_samples * 2):
            result["details"] = "A/B observation window still running"
            return result

        # This is a research recommendation only. Even high scores cannot
        # set an active model or mutate the registry without an independent
        # authorized promotion receipt / provenance and rollback review.
        mean_a, mean_b = sum_a / count_a, sum_b / count_b
        result["action"] = "review_required"
        result["details"] = (
            "A/B feedback eligible for independent review only; "
            f"observed mean scores: arm A={mean_a:.4f}, arm B={mean_b:.4f}; "
            "no model activation or rollback performed"
        )
        return result

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
    elif args.status:
        print(json.dumps(promoter.status(), indent=2, default=str))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
