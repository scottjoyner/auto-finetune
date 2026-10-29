"""Real registry/feedback unit behavior exercised only in isolated tmp dirs."""
import json
from pathlib import Path

import pytest

from src.config import Config
from src.feedback_loop import FeedbackCollector
from src.model_registry import ModelRegistry


def config(tmp_path):
    return Config(raw={"paths": {"analysis_dir": str(tmp_path)}})


def registry(tmp_path):
    return ModelRegistry(config(tmp_path))


def make_version(r, label, base="Qwen/Qwen3-8B"):
    return r.register(label=label, checkpoint_path="/synthetic/checkpoint",
                      base_model=base, metrics={"quality": .5})


def test_empty_registry_has_no_active_or_ab_state(tmp_path):
    r = registry(tmp_path)
    assert r.list_versions() == []
    assert r.active_version() is None
    assert r.list_ab_tests() == []
    assert r.summary()["ab_tests_running"] == 0


def test_registry_register_persist_reload_and_list(tmp_path):
    r = registry(tmp_path)
    v1 = make_version(r, "baseline")
    v2 = make_version(r, "candidate")
    restored = registry(tmp_path)
    assert restored.get(v1).label == "baseline"
    assert restored.get(v2).label == "candidate"
    assert restored.get(v2).metrics["quality"] == .5
    assert len(restored.list_versions()) == 2
    assert restored.summary()["total_versions"] == 2


def test_registry_non_lfm3_manual_promotion_rejected_persistently(tmp_path):
    r = registry(tmp_path)
    v1 = make_version(r, "baseline")
    v2 = make_version(r, "candidate")
    assert not r.promote(v1)
    assert not r.promote(v2)
    restored = registry(tmp_path)
    assert restored.active_version() is None
    assert restored.get(v1).status == "candidate"
    assert restored.get(v2).status == "candidate"
    assert len(restored.list_versions(status="active")) == 0


def test_registry_denies_lfm3_direct_promotion_across_restarts(tmp_path):
    r = registry(tmp_path)
    old = make_version(r, "original")
    trial = make_version(r, "trial", base="LiquidAI/LFM3-1.7B")
    assert not r.promote(old)
    assert not r.promote(trial)
    restored = registry(tmp_path)
    assert restored.active_version() is None
    assert restored.get(trial).status == "candidate"


def test_registry_rejects_unknown_version_and_metrics(tmp_path):
    r = registry(tmp_path)
    v = make_version(r, "candidate")
    assert r.get("missing") is None
    assert not r.promote("missing")
    assert not r.retire("missing")
    assert not r.update_metrics("missing", {"secret": 1})
    assert r.update_metrics(v, {"performance": .8})
    assert registry(tmp_path).get(v).metrics["performance"] == .8


def test_registry_retire_persists_and_summary(tmp_path):
    r = registry(tmp_path)
    v = make_version(r, "candidate")
    assert r.retire(v)
    assert registry(tmp_path).get(v).status == "retired"
    assert registry(tmp_path).summary()["retired"] == 1


def test_registry_ab_start_and_metric_persistence(tmp_path):
    r = registry(tmp_path)
    a = make_version(r, "baseline")
    b = make_version(r, "challenger")
    ab_id = r.start_ab_test(a, b)
    assert ab_id is not None
    assert r.get_ab_test(ab_id).status == "running"
    assert r.record_ab_metric(ab_id, a, "score", .6)
    assert r.record_ab_metric(ab_id, b, "score", .75)
    restored = registry(tmp_path)
    assert restored.get_ab_test(ab_id).metrics_a["score_count"] == 1
    assert restored.get_ab_test(ab_id).metrics_b["score"] == .75
    assert restored.summary()["ab_tests_running"] == 1
    assert len(restored.list_ab_tests(status="running")) == 1


def test_registry_ab_missing_version_refused(tmp_path):
    r = registry(tmp_path)
    a = make_version(r, "baseline")
    assert r.start_ab_test(a, "unknown") is None


def test_registry_ab_winner_lfm3_persists_uncompleted(tmp_path):
    r = registry(tmp_path)
    a = make_version(r, "baseline")
    b = make_version(r, "candidate", base="LiquidAI/LFM3-1.7B")
    ab_id = r.start_ab_test(a, b)
    assert r.record_ab_metric(ab_id, a, "score", .2)
    assert r.record_ab_metric(ab_id, b, "score", .9)
    assert r.complete_ab_test(ab_id) is None
    state = registry(tmp_path)
    assert state.get_ab_test(ab_id).status == "running"
    assert state.get_ab_test(ab_id).winner is None
    assert state.get(b).status == "testing"
    assert state.active_version() is None


def test_registry_ab_non_lfm3_cannot_activate_winner(tmp_path):
    r = registry(tmp_path)
    a = make_version(r, "baseline")
    b = make_version(r, "candidate")
    ab_id = r.start_ab_test(a, b)
    assert r.record_ab_metric(ab_id, a, "score", .3)
    assert r.record_ab_metric(ab_id, b, "score", .8)
    assert r.complete_ab_test(ab_id) is None
    state = registry(tmp_path)
    assert state.get_ab_test(ab_id).winner is None
    assert state.get_ab_test(ab_id).status == "running"
    assert state.active_version() is None
    assert state.get(a).status == "testing" and state.get(b).status == "testing"


def test_registry_ab_cannot_complete_again(tmp_path):
    r = registry(tmp_path)
    a = make_version(r, "baseline")
    b = make_version(r, "candidate")
    id_ = r.start_ab_test(a, b)
    assert r.complete_ab_test(id_) is None
    assert r.complete_ab_test(id_) is None
    assert registry(tmp_path).get_ab_test(id_).status == "running"


def test_corrupt_registry_fails_non_destructively(tmp_path):
    p = Path(tmp_path) / "model-registry.json"
    p.write_text("{not valid JSON}", encoding="utf-8")
    r = registry(tmp_path)
    assert r.summary()["total_versions"] == 0
    assert p.read_text() == "{not valid JSON}"


def feedback(tmp_path):
    return FeedbackCollector(config(tmp_path))


def test_feedback_empty_summary_and_flagged(tmp_path):
    c = feedback(tmp_path)
    assert c.summary()["total_entries"] == 0
    assert c.get_flagged_sessions() == []


def test_feedback_roundtrip_and_repair_pairs(tmp_path):
    c = feedback(tmp_path)
    c.record("session-a", "LFM3", "say hello", "bad answer",
             score=.2, expected_text="good answer", issues=["low_score"])
    fresh = feedback(tmp_path)
    assert fresh.entries[0].input_text == "say hello"
    assert fresh.entries[0].expected_text == "good answer"
    assert fresh.summary()["low_quality_count"] == 1
    out = Path(tmp_path) / "repairs.jsonl"
    assert fresh.generate_repair_pairs(str(out)) == 1
    assert json.loads(out.read_text().splitlines()[0])["good_output"] == "good answer"


def test_feedback_get_flagged_returns_session_strings_not_objects(tmp_path):
    c = feedback(tmp_path)
    c.record("session-a", "LFM3", "hello", "error occurred", score=.2)
    c.record("session-a", "LFM3", "hello", "error occurred", score=.1)
    c.record("session-b", "LFM3", "hello", "ordinary response", score=.8)
    assert c.get_flagged_sessions() == ["session-a"]
    assert all(isinstance(x, str) for x in c.get_flagged_sessions())


def test_feedback_analysis_does_not_duplicate_issue_labels(tmp_path):
    c = feedback(tmp_path)
    c.record("session-a", "LFM3", "sample", "error happened", score=.2)
    before = c.analyze().issue_distribution
    again = c.analyze().issue_distribution
    assert before == again
    assert again["low_score"] == 1
    assert again["error_in_output"] == 1


@pytest.mark.parametrize("score", [
    -0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, None, "1.0",
])
def test_feedback_rejects_invalid_score_before_any_write(tmp_path, score):
    c = feedback(tmp_path)
    with pytest.raises(ValueError, match="feedback_score_must_be_finite_fraction"):
        c.record("private-session", "LFM3", "SECRET_INPUT_TEXT",
                 "SECRET_OUTPUT_TEXT", score=score)
    assert c.entries == []
    assert not (Path(tmp_path) / "feedback.jsonl").exists()


def test_feedback_records_exact_boundary_scores(tmp_path):
    c = feedback(tmp_path)
    c.record("zero", "LFM3", "hello", "one", score=0)
    c.record("one", "LFM3", "hello", "another response", score=1)
    assert len(feedback(tmp_path).entries) == 2


def test_feedback_prompt_and_response_size_limits(tmp_path):
    c = feedback(tmp_path)
    c.record("long", "LFM3", "A" * 2500, "B" * 2500, score=.4)
    saved = feedback(tmp_path).entries[0]
    assert len(saved.input_text) == len(saved.output_text) == 2000


def test_feedback_no_correction_produces_no_repair_pair(tmp_path):
    c = feedback(tmp_path)
    c.record("short", "LFM3", "hello", "answer", score=.6)
    out = Path(tmp_path) / "empty.jsonl"
    assert c.generate_repair_pairs(str(out)) == 0
    assert out.read_text() == ""


def test_feedback_repetition_detected(tmp_path):
    c = feedback(tmp_path)
    c.record("repeat", "LFM3", "hello", "word " * 20, score=.7)
    summary = c.analyze()
    assert summary.issue_distribution["repetitive"] == 1
    assert summary.repair_pairs == 0


def test_registry_version_ids_unique_for_same_label_same_second(tmp_path, monkeypatch):
    monkeypatch.setattr("src.model_registry.time.time", lambda: 1234567890)
    r = registry(tmp_path)
    a = make_version(r, "repeated")
    b = make_version(r, "repeated")
    assert a != b
    assert registry(tmp_path).summary()["total_versions"] == 2


def test_registry_ab_ids_unique_for_same_clock_tick(tmp_path, monkeypatch):
    monkeypatch.setattr("src.model_registry.time.time", lambda: 1234567890)
    r = registry(tmp_path)
    a = make_version(r, "first")
    b = make_version(r, "second")
    one = r.start_ab_test(a, b)
    two = r.start_ab_test(a, b)
    assert one != two
    assert len(registry(tmp_path).list_ab_tests()) == 2


def test_feedback_entry_ids_unique_on_identical_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr("src.feedback_loop.time.time", lambda: 1234567890)
    c = feedback(tmp_path)
    a = c.record("s1", "LFM3", "first", "response", score=.5)
    b = c.record("s2", "LFM3", "second", "response", score=.5)
    assert a != b
    assert len(feedback(tmp_path).entries) == 2
