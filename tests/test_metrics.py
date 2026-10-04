"""Tests for metrics history, best-model selection and regression detection.

Metrics are what a deployment decision is based on, so the tests here check the
*direction* of each comparison rather than just that a number came out. Two
defects here were actively misleading: perplexity was classified with a
substring test on the metric name, so the worst model was reported as best and a
25x blow-up was reported as "no regression".
"""
from __future__ import annotations

import os

import pytest

from src.metrics import LOWER_IS_BETTER, MetricsTracker, metric_direction


@pytest.fixture
def tracker(tmp_path):
    return MetricsTracker(str(tmp_path / "metrics"))


def _add(tracker, label, model_id, version, ts, **kw):
    return tracker.record(label=label, model_id=model_id, version=version,
                          timestamp=ts, **kw)


# --- metric direction ------------------------------------------------------

def test_perplexity_is_lower_is_better():
    # Regression: "loss" in "eval_perplexity" is False, so it was treated as
    # higher-is-better and get_best returned the worst model.
    assert metric_direction("eval_perplexity") == "lower"


@pytest.mark.parametrize("metric", sorted(LOWER_IS_BETTER))
def test_lower_is_better_set_is_explicit(metric):
    assert metric_direction(metric) == "lower"


@pytest.mark.parametrize("metric", ["tool_exact_match", "tool_partial_match",
                                    "tool_recall", "dataset_size",
                                    "train_samples_per_second"])
def test_higher_is_better_metrics(metric):
    assert metric_direction(metric) == "higher"


# --- get_best --------------------------------------------------------------

def test_get_best_loss_picks_lowest(tracker):
    _add(tracker, "L", "a", 1, 1.0, eval_loss=0.9)
    _add(tracker, "L", "b", 2, 2.0, eval_loss=0.3)
    _add(tracker, "L", "c", 3, 3.0, eval_loss=0.6)
    assert tracker.get_best("L", "eval_loss").model_id == "b"


def test_get_best_perplexity_picks_lowest(tracker):
    _add(tracker, "P", "good", 1, 1.0, eval_perplexity=2.0)
    _add(tracker, "P", "bad", 2, 2.0, eval_perplexity=50.0)
    assert tracker.get_best("P", "eval_perplexity").model_id == "good"


def test_get_best_tool_match_picks_highest(tracker):
    _add(tracker, "T", "a", 1, 1.0, tool_exact_match=0.4)
    _add(tracker, "T", "b", 2, 2.0, tool_exact_match=0.9)
    assert tracker.get_best("T", "tool_exact_match").model_id == "b"


def test_get_best_ignores_none_values(tracker):
    _add(tracker, "L", "none", 1, 1.0, eval_loss=None)
    _add(tracker, "L", "real", 2, 2.0, eval_loss=0.5)
    assert tracker.get_best("L", "eval_loss").model_id == "real"


def test_get_best_returns_none_without_data(tracker):
    assert tracker.get_best("missing", "eval_loss") is None
    _add(tracker, "L", "a", 1, 1.0, eval_loss=None)
    assert tracker.get_best("L", "eval_loss") is None


def test_get_best_is_scoped_to_label(tracker):
    _add(tracker, "A", "a-low", 1, 1.0, eval_loss=0.1)
    _add(tracker, "B", "b-high", 1, 2.0, eval_loss=9.9)
    assert tracker.get_best("A", "eval_loss").model_id == "a-low"


# --- detect_regression -----------------------------------------------------

def test_detects_eval_loss_regression(tracker):
    _add(tracker, "L", "good", 1, 1.0, eval_loss=0.5)
    _add(tracker, "L", "bad", 2, 2.0, eval_loss=0.9)
    is_reg, msg = tracker.detect_regression("L", threshold=0.05, metric="eval_loss")
    assert is_reg is True
    assert "regression detected" in msg


def test_detects_perplexity_regression(tracker):
    # The headline case: 2.0 -> 50.0 was previously reported as "latest is best".
    _add(tracker, "P", "good", 1, 1.0, eval_perplexity=2.0)
    _add(tracker, "P", "bad", 2, 2.0, eval_perplexity=50.0)
    is_reg, msg = tracker.detect_regression("P", threshold=0.05,
                                            metric="eval_perplexity")
    assert is_reg is True, msg
    assert "eval_perplexity=50.0000" in msg


def test_no_regression_when_perplexity_improves(tracker):
    _add(tracker, "P", "bad", 1, 1.0, eval_perplexity=50.0)
    _add(tracker, "P", "good", 2, 2.0, eval_perplexity=2.0)
    is_reg, _ = tracker.detect_regression("P", metric="eval_perplexity")
    assert is_reg is False


def test_detects_tool_match_regression(tracker):
    _add(tracker, "T", "good", 1, 1.0, tool_exact_match=0.9)
    _add(tracker, "T", "bad", 2, 2.0, tool_exact_match=0.4)
    is_reg, _ = tracker.detect_regression("T", metric="tool_exact_match")
    assert is_reg is True


def test_no_regression_within_threshold(tracker):
    _add(tracker, "L", "a", 1, 1.0, eval_loss=0.500)
    _add(tracker, "L", "b", 2, 2.0, eval_loss=0.510)   # 2% worse, under 5%
    is_reg, _ = tracker.detect_regression("L", threshold=0.05, metric="eval_loss")
    assert is_reg is False


def test_regression_ignored_when_latest_is_best(tracker):
    _add(tracker, "L", "old", 1, 1.0, eval_loss=0.9)
    _add(tracker, "L", "new", 2, 2.0, eval_loss=0.1)
    is_reg, msg = tracker.detect_regression("L", metric="eval_loss")
    assert is_reg is False
    assert "latest is best" in msg


def test_regression_insufficient_data(tracker):
    assert tracker.detect_regression("nope") == (False, "insufficient data")


def test_regression_metric_not_available(tracker):
    _add(tracker, "L", "a", 1, 1.0, eval_loss=0.5)
    _add(tracker, "L", "b", 2, 2.0)   # no eval_loss at all
    is_reg, msg = tracker.detect_regression("L", metric="eval_loss")
    assert is_reg is False
    assert msg == "metric not available"


# --- get_history -----------------------------------------------------------

def test_get_history_returns_newest_first(tracker):
    for i in range(10):
        _add(tracker, "L", f"m{i}", i, 100.0 + i, eval_loss=1.0)
    hist = tracker.get_history("L", limit=5)
    # Regression: sorted descending then sliced [-limit:] returned the oldest.
    assert [m.version for m in hist] == [9, 8, 7, 6, 5]


def test_get_history_limit_exceeding_count(tracker):
    for i in range(3):
        _add(tracker, "L", f"m{i}", i, 100.0 + i, eval_loss=1.0)
    assert len(tracker.get_history("L", limit=50)) == 3


def test_get_history_filters_by_label(tracker):
    _add(tracker, "A", "a", 1, 1.0, eval_loss=1.0)
    _add(tracker, "B", "b", 1, 2.0, eval_loss=1.0)
    assert [m.label for m in tracker.get_history("A")] == ["A"]


def test_get_latest_picks_newest(tracker):
    _add(tracker, "L", "old", 1, 100.0, eval_loss=0.9)
    _add(tracker, "L", "new", 2, 200.0, eval_loss=0.1)
    assert tracker.get_latest("L").model_id == "new"


def test_get_latest_none_for_unknown_label(tracker):
    assert tracker.get_latest("nope") is None


# --- compare_versions ------------------------------------------------------

def test_compare_defaults_to_latest_two(tracker):
    _add(tracker, "L", "v1", 1, 1.0, eval_loss=0.8)
    _add(tracker, "L", "v2", 2, 2.0, eval_loss=0.4)
    out = tracker.compare_versions("L")
    assert out["v1"] == "v1" and out["v2"] == "v2"
    assert out["metrics"]["eval_loss"]["delta"] == pytest.approx(-0.4)


def test_compare_explicit_versions(tracker):
    _add(tracker, "L", "v1", 1, 1.0, eval_loss=0.8)
    _add(tracker, "L", "v2", 2, 2.0, eval_loss=0.4)
    _add(tracker, "L", "v3", 3, 3.0, eval_loss=0.2)
    out = tracker.compare_versions("L", v1=1, v2=3)
    assert out["metrics"]["eval_loss"]["delta"] == pytest.approx(-0.6)


def test_compare_needs_two_versions(tracker):
    _add(tracker, "L", "v1", 1, 1.0, eval_loss=0.8)
    assert "error" in tracker.compare_versions("L")


def test_compare_unknown_version_errors(tracker):
    _add(tracker, "L", "v1", 1, 1.0, eval_loss=0.8)
    _add(tracker, "L", "v2", 2, 2.0, eval_loss=0.4)
    assert tracker.compare_versions("L", v1=1, v2=99)["error"] == "version not found"


def test_compare_skips_missing_metrics(tracker):
    _add(tracker, "L", "v1", 1, 1.0, eval_loss=0.8)
    _add(tracker, "L", "v2", 2, 2.0, eval_loss=0.4, tool_exact_match=0.5)
    out = tracker.compare_versions("L")
    assert "eval_loss" in out["metrics"]
    assert "tool_exact_match" not in out["metrics"], "reported a metric only one side has"


def test_compare_handles_zero_baseline(tracker):
    _add(tracker, "L", "v1", 1, 1.0, tool_exact_match=0.0)
    _add(tracker, "L", "v2", 2, 2.0, tool_exact_match=0.5)
    out = tracker.compare_versions("L")
    assert out["metrics"]["tool_exact_match"]["pct_change"] == 0


# --- summary ---------------------------------------------------------------

def test_summary_aggregates(tracker):
    _add(tracker, "L", "a", 1, 1.0, eval_loss=0.4, tool_exact_match=0.5)
    _add(tracker, "L", "b", 2, 2.0, eval_loss=0.2, tool_exact_match=0.9)
    out = tracker.summary("L")
    assert out["count"] == 2
    assert out["avg_eval_loss"] == pytest.approx(0.3)
    assert out["best_eval_loss"] == pytest.approx(0.2)
    assert out["best_tool_exact"] == pytest.approx(0.9)


def test_summary_empty(tracker):
    assert tracker.summary("nope") == {"count": 0}


# --- persistence -----------------------------------------------------------

def test_history_survives_reload(tmp_path):
    t = MetricsTracker(str(tmp_path / "m"))
    _add(t, "L", "a", 1, 1.0, eval_loss=0.5)
    _add(t, "L", "b", 2, 2.0, eval_loss=0.4)
    assert len(MetricsTracker(str(tmp_path / "m")).metrics) == 2


def test_save_leaves_no_temp_file(tracker):
    _add(tracker, "L", "a", 1, 1.0, eval_loss=0.5)
    leftovers = [p for p in os.listdir(os.path.dirname(tracker.metrics_path))
                 if p.endswith(".tmp")]
    assert leftovers == [], f"atomic write left temp files: {leftovers}"


def test_corrupt_history_fails_loudly(tmp_path):
    # Silently starting from an empty history would make every later regression
    # check answer "insufficient data" -- the dangerous kind of quiet.
    d = tmp_path / "m"
    os.makedirs(str(d))
    (d / "training-metrics.json").write_text('[{"model_id": "a"')
    with pytest.raises(RuntimeError, match="corrupt metrics history"):
        MetricsTracker(str(d))


# --- CLI -------------------------------------------------------------------

def _cli_cfg(tmp_path):
    from src.config import Config
    d = tmp_path / "analysis"
    os.makedirs(str(d), exist_ok=True)
    return Config(raw={"paths": {"analysis_dir": str(d)}})


@pytest.mark.parametrize("flag,val", [
    ("--version", "abc"), ("--loss", "abc"), ("--eval-loss", "abc"),
    ("--eval-perplexity", "abc"), ("--tool-exact", "abc"),
    ("--dataset-size", "abc"), ("--runtime", "abc"),
])
def test_metrics_record_rejects_non_numeric(tmp_path, flag, val, capsys):
    # These each raised an uncaught ValueError traceback before.
    from src.metrics import main
    rc = main(_cli_cfg(tmp_path), ["cli", "metrics-record", "--label=x",
                                  f"{flag}={val}"])
    assert rc == 2
    assert "[error]" in capsys.readouterr().out


def test_metrics_record_requires_label(tmp_path, capsys):
    from src.metrics import main
    assert main(_cli_cfg(tmp_path), ["cli", "metrics-record"]) == 2
    assert "requires --label" in capsys.readouterr().out


def test_metrics_record_persists_supplied_fields(tmp_path):
    from src.metrics import main
    cfg = _cli_cfg(tmp_path)
    rc = main(cfg, ["cli", "metrics-record", "--label=combined", "--version=3",
                    "--eval-loss=0.42", "--tool-exact=0.85",
                    "--eval-perplexity=2.5"])
    assert rc == 0
    from src.metrics import MetricsTracker
    t = MetricsTracker(os.path.join(cfg.path("analysis_dir"), "metrics"))
    entry = t.metrics[-1]
    assert entry.label == "combined"
    assert entry.version == 3
    assert entry.eval_loss == pytest.approx(0.42)
    assert entry.tool_exact_match == pytest.approx(0.85)
    # Regression: this had no CLI flag, so the metric could not be recorded.
    assert entry.eval_perplexity == pytest.approx(2.5)


def test_metrics_record_model_id_default(tmp_path):
    from src.metrics import main
    cfg = _cli_cfg(tmp_path)
    main(cfg, ["cli", "metrics-record", "--label=combined", "--version=7"])
    from src.metrics import MetricsTracker
    t = MetricsTracker(os.path.join(cfg.path("analysis_dir"), "metrics"))
    assert t.metrics[-1].model_id == "toolcall-v5-3b-combined-v7"


def test_metrics_summary_and_history_commands(tmp_path, capsys):
    from src.metrics import main
    cfg = _cli_cfg(tmp_path)
    main(cfg, ["cli", "metrics-record", "--label=L", "--version=1",
               "--eval-loss=0.5"])
    main(cfg, ["cli", "metrics-record", "--label=L", "--version=2",
               "--eval-loss=0.4"])
    assert main(cfg, ["cli", "metrics-history", "--label=L"]) == 0
    out = capsys.readouterr().out
    hist = out[out.index("[metrics-history]"):]
    assert "2 entries" in hist
    # Newest first, which is the point of the get_history fix.
    assert hist.index("toolcall-v5-3b-L-v2") < hist.index("toolcall-v5-3b-L-v1")
    assert main(cfg, ["cli", "metrics-summary"]) == 0
    assert main(cfg, ["cli", "metrics-regression", "--label=L"]) == 0
