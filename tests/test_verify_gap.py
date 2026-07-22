"""CPU tests for src/verify_gap.py (benchmark ceiling diagnosis)."""
from src.verify_gap import categorize


def _r(ok, reason="", checks=None):
    return {"task_id": "t", "ok": ok, "reason": reason,
            "bucket": "debug", "checks": checks or []}


def test_passing_excluded():
    res = categorize([_r(True)])
    assert res["n_failing"] == 0


def test_categories():
    rows = [
        _r(False, "source session not found"),
        _r(False, "x", [{"ok": False, "detail": "unsupported check kind: foo"}]),
        _r(False, "x", [{"ok": False, "detail": "file not found: /etc/x"}]),
        _r(False, "x", [{"ok": False, "detail": "snippet not in x"}]),
    ]
    res = categorize(rows)
    assert res["n_failing"] == 4
    assert res["counts"] == {
        "no_source": 1, "unsupported_kind": 1,
        "file_not_materialized": 1, "snippet_missing": 1,
    }


def test_counts_rollup():
    rows = [_r(False, "x", [{"ok": False, "detail": "file not found: a"}])] * 3
    res = categorize(rows)
    assert res["n_failing"] == 3
    assert res["counts"]["file_not_materialized"] == 3
