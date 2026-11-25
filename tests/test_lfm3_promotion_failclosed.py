"""No production paths or databases: mocked registry and promotion safety."""
from unittest.mock import Mock
import time

from src.auto_promote import AutoPromoter
from src.model_registry import ABTest, ModelRegistry, ModelVersion


def version(name, base="Qwen/Qwen3-8B"):
    return ModelVersion(version_id=name, label=name, checkpoint_path="/fictional/checkpoint",
                        base_model=base, created_at=1, status="candidate")


def stub_registry(*versions):
    r = ModelRegistry.__new__(ModelRegistry)
    r._versions = {v.version_id: v for v in versions}
    r._ab_tests = {}
    r._save = Mock()
    return r


def stub_promoter(active=None, ab_test="ab-1"):
    p = AutoPromoter.__new__(AutoPromoter)
    p.registry = Mock()
    p.registry.register.return_value = "candidate-1"
    p.registry.active_version.return_value = active
    p.registry.start_ab_test.return_value = ab_test
    p.registry.promote.return_value = True
    p.alerts = Mock()
    p.state = {"active_ab_test": None, "last_promotion": 0, "last_rollback": 0}
    p._save_state = Mock()
    return p


def test_registry_rejects_lfm3_direct_promotion_without_altering_incumbent():
    prior = version("prior")
    prior.status = "active"
    candidate = version("candidate", base="LiquidAI/LFM3-1.7B")
    r = stub_registry(prior, candidate)
    assert r.promote("candidate") is False
    assert prior.status == "active"
    assert candidate.status == "candidate"
    r._save.assert_not_called()


def test_registry_casefolded_lfm3_guard():
    candidate = version("candidate", base="other/LfM3-1.7B")
    r = stub_registry(candidate)
    assert r.promote("candidate") is False
    r._save.assert_not_called()


def test_registry_unknown_promotion_does_not_change_state():
    r = stub_registry(version("prior"))
    assert r.promote("unknown") is False
    r._save.assert_not_called()


def test_existing_non_lfm3_manual_promotion_remains_compatible():
    prior = version("prior")
    prior.status = "active"
    candidate = version("candidate")
    r = stub_registry(prior, candidate)
    assert r.promote("candidate") is True
    assert prior.status == "retired"
    assert candidate.status == "active"
    r._save.assert_called_once()


def test_lfm3_ab_winner_not_completed_or_promoted():
    prior = version("prior")
    candidate = version("candidate", base="LiquidAI/LFM3-1.7B")
    r = stub_registry(prior, candidate)
    ab = ABTest(test_id="ab-1", version_a="prior", version_b="candidate",
                start_time=time.time() - 3600,
                metrics_a={"score": 5, "score_count": 10},
                metrics_b={"score": 9, "score_count": 10})
    r._ab_tests["ab-1"] = ab
    assert r.complete_ab_test("ab-1") is None
    assert ab.status == "running"
    assert ab.winner is None and ab.end_time is None
    assert candidate.status == "candidate"
    r._save.assert_not_called()


def test_non_lfm3_ab_completion_still_supported():
    prior = version("prior")
    cand = version("candidate")
    r = stub_registry(prior, cand)
    ab = ABTest(test_id="ab-1", version_a="prior", version_b="candidate",
                start_time=time.time() - 3600,
                metrics_a={"score": 5, "score_count": 10},
                metrics_b={"score": 8, "score_count": 10})
    r._ab_tests["ab-1"] = ab
    assert r.complete_ab_test("ab-1") == "candidate"
    assert ab.status == "completed" and ab.winner == "candidate"
    assert cand.status == "active" and prior.status == "retired"


def test_autopromote_lfm3_is_candidate_only_with_active_version():
    p = stub_promoter(active=version("prior"))
    assert p.register_and_test("trial", "/fake/model", "LiquidAI/LFM3-1.7B") == "candidate-1"
    p.registry.register.assert_called_once()
    p.registry.promote.assert_not_called()
    p.registry.start_ab_test.assert_not_called()
    p.alerts.fire.assert_not_called()


def test_autopromote_lfm3_is_candidate_only_without_active_version():
    p = stub_promoter(active=None)
    assert p.register_and_test("trial", "/fake/model", "liquidAI/lfm3-1.7b") == "candidate-1"
    p.registry.promote.assert_not_called()
    p.registry.start_ab_test.assert_not_called()


def test_autopromote_non_lfm3_cannot_bootstrap_without_review():
    p = stub_promoter(active=None)
    assert p.register_and_test("trial", "/fake/model", "Qwen/Qwen3-8B") == "candidate-1"
    p.registry.promote.assert_not_called()
    p.registry.start_ab_test.assert_not_called()


def test_autopromote_ab_start_failure_stays_candidate():
    p = stub_promoter(active=version("prior"), ab_test=None)
    assert p.register_and_test("trial", "/fake/model", "Qwen/Qwen3-8B") == "candidate-1"
    p.registry.promote.assert_not_called()


def test_normal_ab_start_still_records_state_without_promotion():
    p = stub_promoter(active=version("prior"), ab_test="ab-1")
    assert p.register_and_test("trial", "/fake/model", "Qwen/Qwen3-8B") == "candidate-1"
    assert p.state["active_ab_test"] == "ab-1"
    p.registry.promote.assert_not_called()
    p._save_state.assert_called_once()


def test_no_active_ab_check_is_read_only():
    p = stub_promoter()
    assert p.check_promotion() == {"action": "none", "details": "No active A/B test"}
    p.registry.complete_ab_test.assert_not_called()
