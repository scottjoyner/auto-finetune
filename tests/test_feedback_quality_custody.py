"""Synthetic feedback-quality custody tests. No real inference or user data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import feedback_loop as feedback


class Config:
    def __init__(self, path):
        self.path_root = path
    def path(self, name):
        assert name == "analysis_dir"
        return str(self.path_root)


def collector(tmp_path):
    return feedback.FeedbackCollector(Config(tmp_path))


def record(obj, session="fake-session", score=0.5, output="synthetic answer",
           expected=None, issues=None):
    return obj.record(session_id=session, model_version="local-research",
                      input_text="synthetic prompt", output_text=output,
                      score=score, expected_text=expected, issues=issues)


def test_initial_registry_has_no_real_feedback(tmp_path):
    obj = collector(tmp_path)
    assert obj.entries == []
    assert obj.analyze().total_entries == 0
    assert obj.get_flagged_sessions() == []
    assert obj.summary()["total_entries"] == 0


def test_record_persists_and_reload_respects_text_limit(tmp_path):
    obj = collector(tmp_path)
    eid = obj.record(
        session_id="S", model_version="research",
        input_text="q" * 2500, output_text="a" * 2500,
        expected_text="b" * 2500, score=.8,
        metadata={"source": "synthetic"}, issues=["format"],
    )
    assert eid.startswith("fb-")
    reloaded = collector(tmp_path)
    e = reloaded.entries[0]
    assert len(e.input_text) == 2000
    assert len(e.output_text) == 2000
    assert len(e.expected_text) == 2000
    assert e.metadata == {"source": "synthetic"}
    assert e.issues == ["format"]
    assert e.score == pytest.approx(.8)


@pytest.mark.parametrize("invalid", [
    float("nan"), float("inf"), float("-inf"), -1, 1.1, None, "0.8", True,
])
def test_invalid_scores_cannot_enter_promotion_feedback(tmp_path, invalid):
    obj = collector(tmp_path)
    with pytest.raises(ValueError, match="feedback_score_out_of_range"):
        record(obj, score=invalid)
    assert obj.entries == []
    assert not (tmp_path / "feedback.jsonl").exists()


def test_score_boundaries_are_accepted(tmp_path):
    obj = collector(tmp_path)
    assert record(obj, session="zero", score=0)
    assert record(obj, session="one", score=1)
    assert len(obj.entries) == 2


@pytest.mark.parametrize(("output", "score", "expected_issue"), [
    ("oops", .9, "too_short"),
    ("Everything failed with an exception", .9, "error_in_output"),
    ("same " * 40, .9, "repetitive"),
    ("clean output", .1, "low_score"),
    ("A" * 5001, .9, "too_long"),
])
def test_detect_issues_once_only(tmp_path, output, score, expected_issue):
    obj = collector(tmp_path)
    e = feedback.FeedbackEntry(
        "e", "s", "m", "prompt", output, score=score)
    first = obj._detect_issues(e)
    assert expected_issue in first
    e.issues = first
    second = obj._detect_issues(e)
    assert first == second
    assert len(first) == len(set(first))


def test_preset_issues_not_duplicated_during_repeated_analysis(tmp_path):
    obj = collector(tmp_path)
    record(obj, "S", .1, "failed", issues=["low_score", "low_score"])
    first = obj.analyze()
    second = obj.analyze()
    assert first.issue_distribution == second.issue_distribution
    assert first.issue_distribution["low_score"] == 1
    assert first.low_quality_count == second.low_quality_count == 1


def test_analysis_stored_summary_and_repair_count(tmp_path):
    obj = collector(tmp_path)
    record(obj, "bad", .1, "failed", expected="corrected response")
    record(obj, "good", .9, "correct enough", expected=None)
    r = obj.analyze()
    assert r.total_entries == 2
    assert r.avg_score == pytest.approx(.5)
    assert r.low_quality_count == 1
    assert r.repair_pairs == 1
    assert "bad" in r.flagged_sessions
    assert "good" not in r.flagged_sessions
    saved = json.loads((tmp_path / "feedback-summary.json").read_text())
    assert saved["repair_pairs"] == 1


def test_flagged_returns_distinct_session_id_strings(tmp_path):
    obj = collector(tmp_path)
    record(obj, "bad", .1, "failed")
    record(obj, "bad", .2, "failed again")
    record(obj, "good", .9, "a good example")
    flagged = obj.get_flagged_sessions()
    assert flagged == ["bad"]
    assert all(isinstance(s, str) for s in flagged)


def test_error_output_is_flagged_even_at_high_score(tmp_path):
    obj = collector(tmp_path)
    record(obj, "error-session", .95, "Traceback: synthetic")
    assert obj.get_flagged_sessions() == ["error-session"]


def test_generate_repair_pairs_only_for_wrong_expected_output(tmp_path):
    obj = collector(tmp_path)
    record(obj, "wrong", .3, "old text", expected="new text")
    record(obj, "match", .9, "same text", expected="same text")
    record(obj, "notarget", .2, "bad text", expected=None)
    target = tmp_path / "isolated" / "repairs.jsonl"
    assert obj.generate_repair_pairs(str(target)) == 1
    rows = [json.loads(x) for x in target.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["good_output"] == "new text"
    assert rows[0]["bad_output"] == "old text"
    assert rows[0]["score"] == .3


def test_corrupt_feedback_registry_is_not_read_as_valid_samples(tmp_path):
    p = tmp_path / "feedback.jsonl"
    p.write_text("NOT-JSON\n")
    obj = collector(tmp_path)
    assert obj.entries == []
    assert obj.summary()["total_entries"] == 0


def test_cli_analyze_and_flagged_use_only_disposable_registry(
        tmp_path, monkeypatch, capsys):
    obj = collector(tmp_path)
    record(obj, "fake-bad", .1, "failed")
    monkeypatch.setattr(feedback, "load", lambda: Config(tmp_path))
    assert feedback.main(["--analyze", "--flagged"]) == 0
    out = capsys.readouterr().out
    assert "Total: 1" in out
    assert "fake-bad" in out


def test_cli_repair_pairs_and_help(tmp_path, monkeypatch, capsys):
    obj = collector(tmp_path)
    record(obj, "session", .3, "old", expected="new")
    monkeypatch.setattr(feedback, "load", lambda: Config(tmp_path))
    target = tmp_path / "subdir" / "repairs.jsonl"
    assert feedback.main(["--repair-pairs", str(target)]) == 0
    assert target.is_file()
    assert "Generated 1 repair pairs" in capsys.readouterr().out
    assert feedback.main([]) == 0
    assert "Feedback loop analysis" in capsys.readouterr().out
