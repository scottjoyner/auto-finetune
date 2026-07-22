"""Tests for auto_balance, dataset_version, and cost modules."""
import json
import os
import tempfile

import pytest


class TestAutoBalance:
    """Tests for automatic data balancing."""

    def test_balance_buckets(self):
        from src.auto_balance import balance_buckets

        bucket_map = {
            "s1": {"bucket": "file-edit", "keep": True},
            "s2": {"bucket": "file-edit", "keep": True},
            "s3": {"bucket": "debug", "keep": True},
            "s4": {"bucket": "debug", "keep": True},
            "s5": {"bucket": "debug", "keep": True},
            "s6": {"bucket": "reasoning", "keep": True},
        }

        balanced = balance_buckets(bucket_map, target_size=10, seed=42)

        assert "file-edit" in balanced
        assert "debug" in balanced
        assert "reasoning" in balanced
        # All buckets should have sessions
        assert len(balanced["file-edit"]) == 2
        assert len(balanced["debug"]) >= 1
        assert len(balanced["reasoning"]) == 1

    def test_compute_balance_stats(self):
        from src.auto_balance import compute_balance_stats

        bucket_map = {
            "s1": {"bucket": "file-edit", "source": "opencode", "difficulty": "easy", "keep": True},
            "s2": {"bucket": "debug", "source": "hermes", "difficulty": "hard", "keep": False},
        }

        stats = compute_balance_stats(bucket_map)

        assert stats["total"] == 2
        assert stats["by_bucket"]["file-edit"] == 1
        assert stats["dropped"] == 1

    def test_bucket_weights(self):
        from src.auto_balance import BUCKET_WEIGHTS

        assert "file-edit" in BUCKET_WEIGHTS
        assert "debug" in BUCKET_WEIGHTS
        assert BUCKET_WEIGHTS["multi-file-refactor"] > BUCKET_WEIGHTS["reasoning"]


class TestDatasetVersion:
    """Tests for dataset versioning."""

    def test_versioner_create(self):
        from src.dataset_version import DatasetVersioner

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test dataset
            dataset_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(dataset_dir)
            with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
                f.write('{"text": "hello"}\n')
                f.write('{"text": "world"}\n')

            versioner = DatasetVersioner(tmpdir)
            version = versioner.create_version(
                dataset_dir, "test", description="test dataset",
            )

            assert version.label == "test"
            assert version.num_examples == 2
            assert version.content_hash

    def test_versioner_list(self):
        from src.dataset_version import DatasetVersioner
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(dataset_dir)

            versioner = DatasetVersioner(tmpdir)

            for i in range(3):
                v = versioner.create_version(dataset_dir, "test")
                assert v.version_id.startswith("v")
                time.sleep(0.1)  # Ensure different timestamps

            versions = versioner.list_versions("test")
            assert len(versions) == 3

    def test_versioner_diff(self):
        from src.dataset_version import DatasetVersioner
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(dataset_dir)

            versioner = DatasetVersioner(tmpdir)

            v1 = versioner.create_version(dataset_dir, "test")
            time.sleep(0.1)

            # Modify dataset
            with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
                f.write('{"text": "modified"}\n')

            v2 = versioner.create_version(dataset_dir, "test")

            diff = versioner.diff(v1.version_id, v2.version_id)

            assert not diff["same_content"]
            assert diff["num_examples"]["v1"] == 0
            assert diff["num_examples"]["v2"] == 1


class TestCost:
    """Tests for cost tracking."""

    def test_cost_tracker_record(self):
        from src.cost import CostTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(tmpdir)

            entry = tracker.record(
                label="combined",
                training_hours=8.5,
                eval_hours=1.5,
                gpu_name="Radeon 890M",
            )

            assert entry.label == "combined"
            assert entry.total_hours == 10.0

    def test_cost_tracker_summary(self):
        from src.cost import CostTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(tmpdir)

            tracker.record(label="ssd", training_hours=4.0)
            tracker.record(label="combined", training_hours=8.0)
            tracker.record(label="combined", training_hours=6.0)

            summary = tracker.get_summary()
            assert summary["count"] == 3
            assert summary["total_hours"] == 18.0

    def test_cost_by_label(self):
        from src.cost import CostTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(tmpdir)

            tracker.record(label="ssd", training_hours=4.0)
            tracker.record(label="combined", training_hours=8.0)

            by_label = tracker.get_by_label()
            assert "ssd" in by_label
            assert "combined" in by_label
            assert by_label["ssd"]["runs"] == 1
            assert by_label["combined"]["runs"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
