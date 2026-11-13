"""Synthetic-only model A/B results and manual promotion boundary tests.

All durable writes target pytest tmp_path. No remote inference, GPU, real model
checkpoint, production registry, token or external service is used.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from src import auto_promote as promo
from src import model_registry as registry


class Config:
    def __init__(self, root):
        self.root = root

    def path(self, name):
        assert name == "analysis_dir"
        return str(self.root)


def make_registry(tmp_path):
    reg = registry.ModelRegistry(Config(tmp_path))
    return reg


def with_versions(tmp_path):
    reg = make_registry(tmp_path)
    incumbent = reg.register("incumbent", "synthetic-a", "fake-base")
    assert reg.promote(incumbent)
    candidate = reg.register("candidate", "synthetic-b", "fake-base")
    return reg, incumbent, candidate


def make_promoter(reg, test_id=None):
    obj = object.__new__(promo.AutoPromoter)
    obj.registry = reg
    obj.config = promo.PromotionConfig()
    obj.state = {"active_ab_test": test_id, "last_promotion": 0, "last_rollback": 0}
    obj.alerts = SimpleNamespace(fire=lambda *a, **k: None)
    obj.feedback = SimpleNamespace(
        record=lambda *a, **k: None,
        summary=lambda: {"samples": 0},
    )
    snapshots = []
    obj._save_state = lambda: snapshots.append(dict(obj.state))
    return obj, snapshots


def complete_ready_test(reg, incumbent, candidate, a=0.7, b=0.9,
                        counts=(50, 50)):
    test_id = reg.start_ab_test(incumbent, candidate)
    assert test_id
    ab = reg.get_ab_test(test_id)
    ab.metrics_a = {"score_count": counts[0], "score": a * counts[0]}
    ab.metrics_b = {"score_count": counts[1], "score": b * counts[1]}
    return test_id, ab


def test_registration_is_only_candidate_and_never_auto_active(tmp_path):
    reg = make_registry(tmp_path)
    promoter, _ = make_promoter(reg)
    vid = promoter.register_and_test("new", "synthetic", "base")
    assert reg.active_version() is None
    assert reg.get(vid).status == "candidate"


def test_ab_start_never_demotes_current_active(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    test_id = reg.start_ab_test(incumbent, candidate)
    assert test_id
    assert reg.active_version().version_id == incumbent
    assert reg.get(incumbent).status == "active"
    assert reg.get(candidate).status == "testing"


def test_new_candidate_starts_test_without_promotion(tmp_path):
    reg, incumbent, _ = with_versions(tmp_path)
    promoter, snapshots = make_promoter(reg)
    vid = promoter.register_and_test("new", "synthetic-c", "base")
    assert promoter.state["active_ab_test"]
    assert snapshots[-1]["active_ab_test"]
    assert reg.active_version().version_id == incumbent
    assert reg.get(vid).status == "testing"


def test_ab_start_failure_does_not_imply_promote(tmp_path, monkeypatch):
    reg, incumbent, _ = with_versions(tmp_path)
    monkeypatch.setattr(reg, "start_ab_test", lambda *a: None)
    promoter, _ = make_promoter(reg)
    vid = promoter.register_and_test("new", "synthetic-c", "base")
    assert reg.active_version().version_id == incumbent
    assert reg.get(vid).status == "candidate"


def test_ab_complete_records_winner_but_not_release(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    test_id, _ = complete_ready_test(reg, incumbent, candidate)
    assert reg.complete_ab_test(test_id) == candidate
    assert reg.active_version().version_id == incumbent
    assert reg.get(incumbent).status == "active"
    assert reg.get(candidate).status == "candidate"
    assert reg.get_ab_test(test_id).status == "completed"
    assert reg.get_ab_test(test_id).winner == candidate
    persisted = json.loads((tmp_path / "model-registry.json").read_text())
    assert any(a["test_id"] == test_id and a["winner"] == candidate
               for a in persisted["ab_tests"])


def test_manual_explicit_promotion_still_separate_action(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    test_id, _ = complete_ready_test(reg, incumbent, candidate)
    reg.complete_ab_test(test_id)
    assert reg.active_version().version_id == incumbent
    assert reg.promote(candidate) is True  # deliberate, isolated fixture action
    assert reg.active_version().version_id == candidate
    assert reg.get(incumbent).status == "retired"


def test_check_ready_for_review_does_not_switch_active(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    test_id, _ = complete_ready_test(reg, incumbent, candidate)
    obj, snapshots = make_promoter(reg, test_id)
    action = obj.check_promotion()
    assert action["action"] == "ready_for_review"
    assert "manual release decision" in action["details"]
    assert obj.state["active_ab_test"] is None
    assert obj.state["last_promotion"] == 0
    assert obj.state["last_rollback"] == 0
    assert obj.state["last_review"] > 0
    assert snapshots
    assert reg.active_version().version_id == incumbent
    assert reg.get(candidate).status == "candidate"


@pytest.mark.parametrize(("scores", "expected"), [
    ((0.9, 0.7), "incumbent"),
    ((0.3, 0.5), "minimum score"),
])
def test_weak_candidate_is_not_activated(tmp_path, scores, expected):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, _ = complete_ready_test(reg, incumbent, candidate,
                                 a=scores[0], b=scores[1])
    obj, _ = make_promoter(reg, tid)
    response = obj.check_promotion()
    assert response["action"] == "hold"
    assert expected in response["details"]
    assert reg.active_version().version_id == incumbent


@pytest.mark.parametrize("counts", [(0, 100), (10, 20), (10, 90)])
def test_unbalanced_or_insufficient_feedback_does_not_complete(tmp_path, counts):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, ab = complete_ready_test(reg, incumbent, candidate, counts=counts)
    obj, _ = make_promoter(reg, tid)
    result = obj.check_promotion()
    assert result["action"] == "none"
    assert "feedback" in result["details"]
    assert reg.get_ab_test(tid).status == "running"
    assert reg.active_version().version_id == incumbent


def test_duration_requires_time_or_double_sample_count(tmp_path, monkeypatch):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, ab = complete_ready_test(reg, incumbent, candidate, counts=(30, 30))
    obj, _ = make_promoter(reg, tid)
    monkeypatch.setattr(promo.time, "time", lambda: ab.start_time + 3600)
    r = obj.check_promotion()
    assert r["action"] == "none"
    assert "still running" in r["details"]
    assert reg.active_version().version_id == incumbent


@pytest.mark.parametrize(("field", "bad"), [
    ("score_count", -1),
    ("score_count", 1.2),
    ("score", float("nan")),
    ("score", float("inf")),
    ("score", -1),
])
def test_bad_score_or_count_is_a_hold(tmp_path, field, bad):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, ab = complete_ready_test(reg, incumbent, candidate)
    ab.metrics_b[field] = bad
    obj, _ = make_promoter(reg, tid)
    r = obj.check_promotion()
    assert r["action"] == "hold"
    assert reg.active_version().version_id == incumbent
    assert reg.get_ab_test(tid).status == "running"


def test_future_ab_start_timestamp_hold(tmp_path, monkeypatch):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, ab = complete_ready_test(reg, incumbent, candidate)
    ab.start_time += 10000
    obj, _ = make_promoter(reg, tid)
    assert obj.check_promotion()["action"] == "hold"
    assert reg.active_version().version_id == incumbent


def test_completion_error_cannot_be_treated_as_approved(tmp_path, monkeypatch):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, _ = complete_ready_test(reg, incumbent, candidate)
    monkeypatch.setattr(reg, "complete_ab_test", lambda *a: None)
    obj, _ = make_promoter(reg, tid)
    assert obj.check_promotion()["action"] == "hold"
    assert obj.state["active_ab_test"] == tid
    assert reg.active_version().version_id == incumbent


def test_missing_or_nonrunning_test_clears_stale_projection(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    obj, updates = make_promoter(reg, "unknown-test")
    result = obj.check_promotion()
    assert result["action"] == "none"
    assert "not running" in result["details"]
    assert obj.state["active_ab_test"] is None
    assert updates


def test_no_test_does_not_create_release(tmp_path):
    reg = make_registry(tmp_path)
    obj, _ = make_promoter(reg)
    assert obj.check_promotion()["action"] == "none"
    assert reg.active_version() is None


def test_recorder_updates_only_live_test_metrics(tmp_path):
    reg, incumbent, candidate = with_versions(tmp_path)
    tid, _ = complete_ready_test(reg, incumbent, candidate)
    obj, _ = make_promoter(reg, tid)
    seen = []
    obj.feedback = SimpleNamespace(record=lambda **k: seen.append(k))
    obj.record_inference(candidate, "synthetic-session", "fake-input", "fake-output", 0.9)
    assert seen[0]["model_version"] == candidate
    ab = reg.get_ab_test(tid)
    assert ab.metrics_b["score_count"] == 51
    assert reg.active_version().version_id == incumbent


def test_cli_check_reports_pending_review_without_success(monkeypatch, capsys):
    class FakePromoter:
        def __init__(self, cfg): pass
        def check_promotion(self):
            return {"action": "ready_for_review", "details": "manual release decision required"}
    monkeypatch.setattr(promo, "load", lambda: object())
    monkeypatch.setattr(promo, "AutoPromoter", FakePromoter)
    assert promo.main(["--check"]) == 78
    assert "manual release decision" in capsys.readouterr().out
