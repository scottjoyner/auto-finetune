"""Offline edge/failure coverage for deterministic pure helpers in small modules.

Only in-memory inputs are used: no filesystem, network, GPU, DB, or wall clock.
`src/data_freshness.py` has no pure helper surface (both age helpers walk the
filesystem and read the clock, and the staleness rules live inside assess_freshness),
so it is intentionally skipped here and reported as NO DATA.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from src.cost import CostEntry, CostTracker
from src.feedback_loop import FeedbackCollector, FeedbackEntry, FeedbackSummary


# --------------------------------------------------------------------------- #
# src/cost.py — entry container and query helpers (no clock, no I/O)
# --------------------------------------------------------------------------- #

def _entry(label: str = "run", timestamp: float = 1.0, **overrides) -> CostEntry:
    fields = dict(entry_id="e", label=label, timestamp=timestamp,
                  training_hours=1.0)
    fields.update(overrides)
    return CostEntry(**fields)


def _tracker(*entries: CostEntry) -> CostTracker:
    """Build a tracker from memory only; __init__/_load would touch the disk."""
    tracker = object.__new__(CostTracker)
    tracker.entries = list(entries)
    return tracker


def test_cost_entry_optional_defaults_and_asdict():
    entry = _entry()
    assert entry.eval_hours is None
    assert entry.total_hours is None
    assert entry.gpu_name is None
    assert entry.gpu_hours is None
    assert entry.disk_usage_mb is None
    assert entry.checkpoints_mb is None
    assert entry.cost_usd is None
    assert entry.notes is None
    assert entry.run_id is None
    assert asdict(entry)["timestamp"] == 1.0


def test_get_summary_empty_entries():
    assert _tracker().get_summary() == {"count": 0}
    assert _tracker().get_summary(label="missing") == {"count": 0}


def test_get_summary_treats_none_as_zero_and_rounds():
    tracker = _tracker(
        _entry(label="a", training_hours=1.0, eval_hours=None,
               total_hours=None, cost_usd=None),
        _entry(label="b", training_hours=2.005, eval_hours=0.5,
               total_hours=2.505, cost_usd=1.005),
    )
    summary = tracker.get_summary()
    assert summary["count"] == 2
    assert summary["total_hours"] == 2.5  # 0 + 2.505 -> round(2.505, 2) == 2.5 in binary float
    assert summary["training_hours"] == 3.0  # 1.0 + 2.005 -> round(3.005, 2)
    assert summary["eval_hours"] == 0.5
    assert summary["total_cost_usd"] == 1.0
    assert summary["avg_hours_per_run"] == 1.25
    assert set(summary["labels"]) == {"a", "b"}


def test_get_summary_label_filter():
    tracker = _tracker(_entry(label="a", total_hours=4.0, cost_usd=2.0),
                       _entry(label="b", total_hours=10.0, cost_usd=9.0))
    summary = tracker.get_summary("a")
    assert summary["count"] == 1
    assert summary["total_hours"] == 4.0
    assert summary["total_cost_usd"] == 2.0
    assert summary["labels"] == ["a"]


def test_get_summary_unknown_label_has_no_results():
    assert _tracker(_entry(label="a")).get_summary("zzz") == {"count": 0}


def test_get_history_sorts_descending_and_applies_slice_limit():
    tracker = _tracker(_entry(label="a", timestamp=1.0),
                       _entry(label="a", timestamp=3.0),
                       _entry(label="b", timestamp=2.0))
    # Current behavior: sorted newest-first, then [-limit:] selects the tail,
    # i.e. the oldest `limit` entries (in descending order). Characterized
    # as-is; no source change is made in this slice.
    history = tracker.get_history(limit=2)
    assert [e.timestamp for e in history] == [2.0, 1.0]
    assert [e.entry_id for e in tracker.get_history(limit=10)] == ["e", "e", "e"]


def test_get_history_filters_label_before_limit():
    tracker = _tracker(_entry(label="a", timestamp=1.0),
                       _entry(label="b", timestamp=5.0),
                       _entry(label="a", timestamp=3.0))
    history = tracker.get_history(label="a", limit=1)
    assert len(history) == 1
    assert history[0].timestamp == 1.0  # tail of the descending, filtered list
    assert history[0].label == "a"


def test_get_history_empty_and_zero_limit_edges():
    assert _tracker().get_history() == []
    # Current slice semantics: [-0:] is the whole list, not an empty result.
    tracker = _tracker(_entry(timestamp=1.0), _entry(timestamp=2.0))
    assert len(tracker.get_history(limit=0)) == 2


def test_get_by_label_groups_counts_totals_and_last_run():
    tracker = _tracker(
        _entry(label="a", timestamp=1.0, total_hours=1.0),
        _entry(label="a", timestamp=4.0, total_hours=None),
        _entry(label="b", timestamp=2.0, total_hours=2.5),
    )
    grouped = tracker.get_by_label()
    assert grouped["a"] == {"runs": 2, "total_hours": 1.0, "last_run": 4.0}
    assert grouped["b"] == {"runs": 1, "total_hours": 2.5, "last_run": 2.0}


def test_get_by_label_empty():
    assert _tracker().get_by_label() == {}


# --------------------------------------------------------------------------- #
# src/feedback_loop.py — entry containers and pure issue detection
# --------------------------------------------------------------------------- #

def _feedback_entry(**overrides) -> FeedbackEntry:
    fields = dict(entry_id="fb", session_id="s", model_version="v1",
                  input_text="in", output_text="a complete answer here",
                  score=0.9)
    fields.update(overrides)
    return FeedbackEntry(**fields)


def _collector() -> FeedbackCollector:
    """Issue detection needs no config, file, or clock; avoid __init__ I/O."""
    return object.__new__(FeedbackCollector)


def test_feedback_entry_mutable_defaults_are_independent():
    first = _feedback_entry()
    second = _feedback_entry()
    first.issues.append("custom")
    first.metadata["k"] = "v"
    assert second.issues == []
    assert second.metadata == {}


def test_feedback_summary_defaults_and_asdict():
    summary = FeedbackSummary()
    assert summary.total_entries == 0
    assert summary.avg_score == 0.0
    assert summary.low_quality_count == 0
    assert summary.issue_distribution == {}
    assert summary.flagged_sessions == []
    assert summary.repair_pairs == 0
    data = asdict(FeedbackSummary(total_entries=2, avg_score=0.25,
                                  issue_distribution={"low_score": 1}))
    assert data["total_entries"] == 2
    assert data["issue_distribution"] == {"low_score": 1}


def test_detect_issues_keeps_preexisting_and_flags_low_score():
    entry = _feedback_entry(score=0.2, issues=["custom"])
    assert _collector()._detect_issues(entry) == ["custom", "low_score"]


def test_detect_issues_score_boundary_is_strict():
    assert _collector()._detect_issues(_feedback_entry(score=0.3)) == []
    assert _collector()._detect_issues(_feedback_entry(score=0.29)) == ["low_score"]


@pytest.mark.parametrize("output,expected", [
    ("short", "too_short"),       # 5 chars, non-empty
    ("x" * 5001, "too_long"),     # strictly above 5000
])
def test_detect_issues_length_edges(output, expected):
    assert _collector()._detect_issues(_feedback_entry(output_text=output)) == [expected]


@pytest.mark.parametrize("output", ["", "1234567890", "x" * 5000])
def test_detect_issues_length_boundaries_do_not_fire(output):
    assert _collector()._detect_issues(_feedback_entry(output_text=output)) == []


def test_detect_issues_repetitive_requires_over_ten_words():
    repetitive = " ".join(["same"] * 11)  # ratio 1/11 < 0.3
    assert _collector()._detect_issues(_feedback_entry(output_text=repetitive)) == ["repetitive"]
    not_repetitive = " ".join(["a", "b", "c", "d"] * 3)  # 12 words, ratio 4/12
    assert _collector()._detect_issues(_feedback_entry(output_text=not_repetitive)) == []


@pytest.mark.parametrize("pattern", ["error", "Exception", "TRACEBACK", "failed", "CaNnot"])
def test_detect_issues_error_patterns_are_case_insensitive(pattern):
    output = f"the tool returned a {pattern} after several retries"
    assert _collector()._detect_issues(_feedback_entry(output_text=output)) == ["error_in_output"]


def test_detect_issues_multiple_error_patterns_flag_once():
    output = "error followed by an exception and a traceback, cannot proceed"
    assert _collector()._detect_issues(_feedback_entry(output_text=output)) == ["error_in_output"]


def test_detect_issues_empty_output_has_no_text_flags():
    assert _collector()._detect_issues(_feedback_entry(output_text="", score=0.8)) == []


def test_detect_issues_combines_flags_in_order():
    entry = _feedback_entry(score=0.0, output_text="error")
    assert _collector()._detect_issues(entry) == ["low_score", "too_short", "error_in_output"]
