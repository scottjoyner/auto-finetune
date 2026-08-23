"""Tests for auto-harvester modules: harvest, deploy, registry, scheduler."""
import json
import os
import tempfile

import pytest


class TestHarvest:
    """Tests for data drift detection and batch planning."""

    def test_harvest_plan_skips_when_no_data(self):
        from src.harvest import HarvestPlan, SourceStats

        plan = HarvestPlan(
            should_harvest=False,
            should_train=False,
            sources=[],
            total_new=0,
            estimated_train_hours=0,
            batch_labels=[],
            reason="no new data",
        )

        assert not plan.should_harvest
        assert not plan.should_train
        assert plan.total_new == 0

    def test_source_stats(self):
        from src.harvest import SourceStats

        s = SourceStats(
            name="opencode",
            db_path="/tmp/test.db",
            total_sessions=100,
            last_modified=1000000,
            db_size_bytes=1024,
            new_sessions=25,
            last_harvest=999000,
            days_since_harvest=1.5,
        )

        assert s.name == "opencode"
        assert s.new_sessions == 25
        assert s.days_since_harvest == 1.5


class TestDeploy:
    """Tests for model deployment."""

    def test_deploy_result(self):
        from src.deploy import DeployResult

        r = DeployResult(
            success=True,
            label="combined",
            target="local",
            deploy_path="/tmp/model",
            message="deployed v1",
            duration_seconds=5.0,
        )

        assert r.success
        assert r.label == "combined"
        assert r.duration_seconds == 5.0

    def test_deployed_model(self):
        from src.deploy import DeployedModel

        m = DeployedModel(
            label="combined",
            source_path="/src/model",
            deploy_path="/dst/model",
            deployed_at=1000000,
            version=1,
            size_bytes=1024000,
            status="active",
            health_check_passed=True,
        )

        assert m.status == "active"
        assert m.health_check_passed

    def test_multi_deploy_result_list(self):
        from src.deploy import DeployResult

        results = [
            DeployResult(True, "c", "node1", "/p", "ok", 1.0),
            DeployResult(True, "c", "node2", "/p", "ok", 1.5),
            DeployResult(False, "c", "node3", "", "fail", 0.5),
        ]

        successful = sum(1 for r in results if r.success)
        assert successful == 2
        assert len(results) == 3


class TestRegistry:
    """Tests for model registry."""

    def test_registry_add_and_get(self):
        from src.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            entry = registry.register(
                label="combined",
                checkpoint_path="/tmp/checkpoint",
                base_model="Qwen/Qwen2.5-7B-Instruct",
            )

            assert entry.label == "combined"
            assert entry.version == 1
            assert entry.model_id == "toolcall-v5-3b-combined-v1"

            # Get latest
            latest = registry.get_latest("combined")
            assert latest is not None
            assert latest.version == 1

            # Add another version
            entry2 = registry.register(
                label="combined",
                checkpoint_path="/tmp/checkpoint2",
            )
            assert entry2.version == 2

            latest = registry.get_latest("combined")
            assert latest.version == 2

    def test_registry_lineage(self):
        from src.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            e1 = registry.register(label="base", checkpoint_path="/m1")
            e2 = registry.register(label="v2", checkpoint_path="/m2", parent_model=e1.model_id)
            e3 = registry.register(label="v3", checkpoint_path="/m3", parent_model=e2.model_id)

            lineage = registry.get_lineage(e3.model_id)
            assert len(lineage) == 3
            assert lineage[0].model_id == e3.model_id
            assert lineage[2].model_id == e1.model_id

    def test_registry_prune(self):
        from src.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            for i in range(5):
                registry.register(label="test", checkpoint_path=f"/m{i}")

            assert len(registry.list_models("test")) == 5

            registry.prune_old_versions("test", keep=2)

            models = registry.list_models("test")
            active = [m for m in models if m.status != "archived"]
            assert len(active) == 2


class TestScheduler:
    """Tests for scheduler."""

    def test_scheduler_state(self):
        from src.scheduler import SchedulerState

        state = SchedulerState(
            last_run=0, last_harvest=0, last_train=0, last_deploy=0,
            runs_completed=0, runs_failed=0, current_phase="idle",
            last_error=None,
        )

        assert state.current_phase == "idle"
        assert state.runs_completed == 0


class TestPlanHarvestDecoupling:
    """Harvest must never be gated by the training-time budget."""

    def test_oversized_source_is_harvested_but_training_deferred(self, monkeypatch):
        from src.harvest import SourceStats, plan_harvest
        from src.config import load

        def fake_stats(cfg):
            return [
                SourceStats(
                    name="opencode", db_path="/tmp/a.db", total_sessions=50,
                    last_modified=1000000, db_size_bytes=1024,
                    new_sessions=50, last_harvest=0, days_since_harvest=9.0,
                ),
                SourceStats(
                    name="hermes", db_path="/tmp/b.db", total_sessions=4079,
                    last_modified=1000000, db_size_bytes=1024,
                    new_sessions=4079, last_harvest=0, days_since_harvest=9.0,
                ),
            ]

        monkeypatch.setattr("src.harvest.get_source_stats", fake_stats)
        cfg = load()
        plan = plan_harvest(cfg)

        # Both labels are queued for cheap CPU extraction...
        assert set(plan.harvest_labels) == {"opencode", "hermes"}
        assert plan.should_harvest
        # ...but the 73h hermes training estimate cannot fit the batch budget.
        assert "hermes" not in plan.batch_labels
        assert plan.batch_labels == ["opencode"]
        # Training still proceeds on whatever fit the budget.
        assert plan.should_train
        assert any("deferred hermes training" in r for r in plan.reason.split("; "))

    def test_small_batches_stay_fully_aligned(self, monkeypatch):
        from src.harvest import SourceStats, plan_harvest
        from src.config import load

        def fake_stats(cfg):
            return [SourceStats(
                name="opencode", db_path="/tmp/a.db", total_sessions=60,
                last_modified=1000000, db_size_bytes=1024,
                new_sessions=60, last_harvest=0, days_since_harvest=2.0,
            )]

        monkeypatch.setattr("src.harvest.get_source_stats", fake_stats)
        plan = plan_harvest(load())

        assert plan.should_harvest and plan.should_train
        assert plan.harvest_labels == plan.batch_labels == ["opencode"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
