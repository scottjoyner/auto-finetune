"""Manual model promotion gate tests use synthetic registry mocks and scratch files."""
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.auto_promote import AutoPromoter, PromotionConfig


class PromotionReviewTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="promotion-review-tests-")
        self.addCleanup(self.scratch.cleanup)
        cfg = MagicMock()
        cfg.path.return_value = self.scratch.name
        self.promoter = AutoPromoter(cfg)
        self.promoter.registry.promote = MagicMock()
        self.promoter.registry.complete_ab_test = MagicMock()
        self.promoter.alerts.fire = MagicMock()
        self.assertTrue(PromotionConfig().require_manual_approval)

    def test_eligible_comparison_stays_manual_review_only(self):
        p = self.promoter
        p.config.min_feedback_samples = 2
        p.config.ab_test_duration_hours = 0.0
        p.state["active_ab_test"] = "synthetic-ab"
        p.registry.get_ab_test = MagicMock(return_value=SimpleNamespace(
            status="running", start_time=time.time()-3600,
            version_a="old", version_b="candidate",
            metrics_a={"score_count": 2}, metrics_b={"score_count": 2}))
        result = p.check_promotion()
        self.assertEqual(result["action"], "review_required")
        p.registry.promote.assert_not_called()
        p.registry.complete_ab_test.assert_not_called()

    def test_imbalanced_feedback_never_promotes(self):
        p = self.promoter
        p.config.min_feedback_samples = 3
        p.state["active_ab_test"] = "synthetic-ab"
        p.registry.get_ab_test = MagicMock(return_value=SimpleNamespace(
            status="running", start_time=time.time()-3600,
            metrics_a={"score_count": 100}, metrics_b={"score_count": 1}))
        self.assertEqual(p.check_promotion()["action"], "none")
        p.registry.complete_ab_test.assert_not_called()

    def test_duration_cannot_be_bypassed_by_one_large_arm(self):
        p = self.promoter
        p.config.min_feedback_samples = 2
        p.config.ab_test_duration_hours = 24.0
        p.state["active_ab_test"] = "synthetic-ab"
        ab = SimpleNamespace(
            status="running", start_time=time.time()-10,
            metrics_a={"score_count": 100}, metrics_b={"score_count": 2})
        p.registry.get_ab_test = MagicMock(return_value=ab)
        result = p.check_promotion()
        self.assertEqual(result["action"], "none")
        self.assertIn("still running", result["details"])
        p.registry.complete_ab_test.assert_not_called()
        ab.metrics_b["score_count"] = 4
        self.assertEqual(p.check_promotion()["action"], "review_required")
        p.registry.complete_ab_test.assert_not_called()

    def test_invalid_counts_and_failed_ab_start_fail_closed(self):
        p = self.promoter
        p.state["active_ab_test"] = "synthetic-ab"
        p.registry.get_ab_test = MagicMock(return_value=SimpleNamespace(
            status="running", start_time=time.time()-3600,
            metrics_a={"score_count": float("nan")},
            metrics_b={"score_count": 100}))
        self.assertEqual(p.check_promotion()["action"], "none")
        p.registry.complete_ab_test.assert_not_called()
        p.registry.register = MagicMock(return_value="new")
        p.registry.active_version = MagicMock(
            return_value=SimpleNamespace(version_id="old"))
        p.registry.start_ab_test = MagicMock(return_value=None)
        self.assertEqual(p.register_and_test(
            "fictional", "/tmp/fictional", "base"), "new")
        p.registry.promote.assert_not_called()
        p.alerts.fire.assert_called_once()

    def test_inference_feedback_records_only_active_ab_metrics(self):
        p = self.promoter
        p.feedback.record = MagicMock()
        p.registry.record_ab_metric = MagicMock()
        p.state["active_ab_test"] = "synthetic-ab"
        p.registry.get_ab_test = MagicMock(return_value=SimpleNamespace(status="running"))
        p.record_inference("model-v1","fictional-session",
                           "synthetic input","synthetic output",0.75)
        p.feedback.record.assert_called_once()
        p.registry.record_ab_metric.assert_called_once_with(
            "synthetic-ab","model-v1","score",0.75)
        p.registry.promote.assert_not_called()

    def test_no_baseline_retains_candidate(self):
        p = self.promoter
        p.registry.register = MagicMock(return_value="synthetic-candidate")
        p.registry.active_version = MagicMock(return_value=None)
        self.assertEqual(p.register_and_test(
            "fictional", "/tmp/nonexistent-synthetic", "base"
        ), "synthetic-candidate")
        p.registry.promote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
