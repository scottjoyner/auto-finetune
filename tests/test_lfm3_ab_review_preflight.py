"""A/B read-only review gate and mock-only registry metric custody tests."""
import math
import time
from unittest.mock import Mock

import pytest

from src.auto_promote import AutoPromoter, PromotionConfig
from src.model_registry import ABTest, ModelRegistry, ModelVersion


def promoter(a=25, b=25, mean_a=.65, mean_b=.7, hours=30, min_feedback=50):
    p = AutoPromoter.__new__(AutoPromoter)
    p.config = PromotionConfig(min_feedback_samples=min_feedback)
    p.state = {"active_ab_test": "ab-1", "last_promotion": 0, "last_rollback": 0}
    p._save_state = Mock()
    p.registry = Mock()
    p.feedback = Mock()
    p.alerts = Mock()
    ab = ABTest(test_id="ab-1", version_a="old", version_b="candidate",
                start_time=time.time()-hours*3600,
                metrics_a={"score_count": a, "score": a*mean_a},
                metrics_b={"score_count": b, "score": b*mean_b})
    p.registry.get_ab_test.return_value = ab
    return p, ab


def version(i, base="Qwen/Qwen3-8B"):
    return ModelVersion(version_id=i, label=i, checkpoint_path="/synthetic",
                        base_model=base, created_at=1)


def registry():
    r = ModelRegistry.__new__(ModelRegistry)
    r._versions = {"old": version("old"), "candidate": version("candidate")}
    r._ab_tests = {"ab-1": ABTest(test_id="ab-1", version_a="old", version_b="candidate",
                                 start_time=time.time()-86400)}
    r._save = Mock()
    return r


def assert_no_activation(p):
    p.registry.complete_ab_test.assert_not_called()
    p.registry.promote.assert_not_called()
    p.registry.record_ab_metric.assert_not_called()
    p.alerts.fire.assert_not_called()
    p._save_state.assert_not_called()


def test_sufficient_valid_ab_feedback_requires_independent_review():
    p, ab = promoter()
    status_before = (ab.status, ab.end_time, ab.winner)
    receipt = p.check_promotion()
    assert receipt["action"] == "review_required"
    assert "arm A=0.6500" in receipt["details"]
    assert "arm B=0.7000" in receipt["details"]
    assert (ab.status, ab.end_time, ab.winner) == status_before
    assert p.state["active_ab_test"] == "ab-1"
    assert_no_activation(p)


def test_no_active_ab_is_read_only():
    p, _ = promoter()
    p.state["active_ab_test"] = None
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


@pytest.mark.parametrize("a,b", [(0, 50), (50, 0), (11, 89), (89, 11)])
def test_one_sided_feedback_cannot_qualify(a, b):
    p, _ = promoter(a=a, b=b)
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


def test_total_count_below_limit_denied():
    p, _ = promoter(a=20, b=20)
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


def test_early_review_with_less_than_double_volume_denied():
    p, _ = promoter(a=30, b=30, hours=1)
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


def test_early_review_with_double_volume_is_still_manual():
    p, _ = promoter(a=50, b=50, hours=1)
    assert p.check_promotion()["action"] == "review_required"
    assert_no_activation(p)


@pytest.mark.parametrize("score_a", [float("nan"), float("inf"), -1, 26.0, True, None, "secret"])
def test_invalid_a_score_denied(score_a):
    p, ab = promoter()
    ab.metrics_a["score"] = score_a
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


@pytest.mark.parametrize("count_a", [float("nan"), 2.0, -3, True, "100", None])
def test_invalid_count_denied(count_a):
    p, ab = promoter()
    ab.metrics_a["score_count"] = count_a
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


def test_future_start_date_denied():
    p, ab = promoter()
    ab.start_time = time.time() + 10000
    assert p.check_promotion()["action"] == "none"
    assert_no_activation(p)


def test_a_b_score_below_threshold_is_still_not_promoted():
    p, _ = promoter(mean_a=.1, mean_b=.2)
    assert p.check_promotion()["action"] == "review_required"
    assert_no_activation(p)


def test_missing_ab_clears_only_obsolete_session_not_registry():
    p, _ = promoter()
    p.registry.get_ab_test.return_value = None
    r = p.check_promotion()
    assert r["action"] == "none"
    assert p.state["active_ab_test"] is None
    p._save_state.assert_called_once()
    p.registry.promote.assert_not_called()
    p.registry.complete_ab_test.assert_not_called()


@pytest.mark.parametrize("metric,value", [
    ("score_count", .8), ("Score", .5), ("e-mail", .5), ("token\nname", .5),
    ("", .5), ("score", -1), ("score", 1.1), ("score", float("nan")),
    ("score", float("inf")), ("score", True), ("score", "0.5"),
    ("latency_ms", float("inf")), ("score", None),
])
def test_registry_rejects_metric_poisoning(metric, value):
    r = registry()
    assert r.record_ab_metric("ab-1", "candidate", metric, value) is False
    assert r._ab_tests["ab-1"].metrics_b == {}
    r._save.assert_not_called()


def test_registry_accepts_bounded_score_and_tracks_count():
    r = registry()
    assert r.record_ab_metric("ab-1", "candidate", "score", .8)
    assert r.record_ab_metric("ab-1", "candidate", "score", .6)
    assert r._ab_tests["ab-1"].metrics_b == {"score": 1.4, "score_count": 2}
    assert r._save.call_count == 2


def test_registry_records_non_score_duration_without_impacting_score_count():
    r = registry()
    assert r.record_ab_metric("ab-1", "old", "latency_ms", 65.0)
    assert r._ab_tests["ab-1"].metrics_a == {"latency_ms": 65.0,
                                            "latency_ms_count": 1}
    assert "score_count" not in r._ab_tests["ab-1"].metrics_a


def test_registry_rejects_unknown_model_without_writing():
    r = registry()
    assert not r.record_ab_metric("ab-1", "rogue", "score", .9)
    r._save.assert_not_called()


def test_registry_rejects_unknown_or_completed_test_without_writing():
    r = registry()
    assert not r.record_ab_metric("missing", "old", "score", .9)
    r._ab_tests["ab-1"].status = "completed"
    assert not r.record_ab_metric("ab-1", "old", "score", .9)
    r._save.assert_not_called()
