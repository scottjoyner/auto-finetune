"""Tests for notify, metrics, and quantize modules."""
import json
import os
import tempfile

import pytest


class TestNotify:
    """Tests for notification system."""

    def test_notification_severity(self):
        from src.notify import EVENT_SEVERITY, Notification

        assert "training_complete" in EVENT_SEVERITY
        assert "deploy_failed" in EVENT_SEVERITY
        assert EVENT_SEVERITY["training_complete"] == "info"
        assert EVENT_SEVERITY["deploy_failed"] == "error"

    def test_notification_creation(self):
        from src.notify import Notification

        n = Notification(
            event="training_complete",
            message="combined model trained",
            timestamp=1000000,
            severity="info",
        )

        assert n.event == "training_complete"
        assert n.severity == "info"


class TestMetrics:
    """Tests for metrics tracking."""

    def test_metrics_tracker_record(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            entry = tracker.record(
                model_id="toolcall-v5-3b-combined-v1",
                label="combined",
                version=1,
                timestamp=1000000,
                train_loss=0.5,
                eval_loss=0.45,
                tool_exact_match=0.82,
            )

            assert entry.label == "combined"
            assert entry.train_loss == 0.5

            # Verify persistence
            tracker2 = MetricsTracker(tmpdir)
            assert len(tracker2.metrics) == 1

    def test_metrics_get_latest(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            tracker.record(
                model_id="v1", label="test", version=1,
                timestamp=1000000, eval_loss=0.5,
            )
            tracker.record(
                model_id="v2", label="test", version=2,
                timestamp=1000001, eval_loss=0.4,
            )

            latest = tracker.get_latest("test")
            assert latest.version == 2

    def test_metrics_get_best(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            tracker.record(
                model_id="v1", label="test", version=1,
                timestamp=1000000, eval_loss=0.5,
            )
            tracker.record(
                model_id="v2", label="test", version=2,
                timestamp=1000001, eval_loss=0.4,
            )

            best = tracker.get_best("test", "eval_loss")
            assert best.version == 2  # lower loss is better

    def test_metrics_detect_regression(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            tracker.record(
                model_id="v1", label="test", version=1,
                timestamp=1000000, eval_loss=0.4,
            )
            tracker.record(
                model_id="v2", label="test", version=2,
                timestamp=1000001, eval_loss=0.6,  # worse
            )

            is_reg, msg = tracker.detect_regression("test")
            assert is_reg
            assert "regression" in msg.lower()

    def test_metrics_compare_versions(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            tracker.record(
                model_id="v1", label="test", version=1,
                timestamp=1000000, eval_loss=0.5, tool_exact_match=0.8,
            )
            tracker.record(
                model_id="v2", label="test", version=2,
                timestamp=1000001, eval_loss=0.4, tool_exact_match=0.85,
            )

            comp = tracker.compare_versions("test")
            assert "v1" in comp
            assert "v2" in comp
            assert "eval_loss" in comp["metrics"]

    def test_metrics_summary(self):
        from src.metrics import MetricsTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)

            for i in range(5):
                tracker.record(
                    model_id=f"v{i}", label="test", version=i,
                    timestamp=1000000 + i, eval_loss=0.5 - i * 0.02,
                )

            summary = tracker.summary("test")
            assert summary["count"] == 5
            assert summary["avg_eval_loss"] is not None


class TestQuantize:
    """Tests for quantization."""

    def test_quantize_result(self):
        from src.quantize import QuantizeResult

        r = QuantizeResult(
            success=True, label="combined",
            source_path="/src", output_path="/dst",
            bits=4, original_size_mb=1000,
            quantized_size_mb=250, compression_ratio=4.0,
            duration_seconds=60.0, message="OK",
        )

        assert r.success
        assert r.compression_ratio == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
