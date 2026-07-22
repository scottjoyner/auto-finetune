"""CPU tests for src/audit.py (train<->benchmark leakage check)."""
from src.audit import _norm, audit_leakage


def _train(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text},
                         {"role": "assistant", "content": "ok"}]}


def test_norm_collapses_noise():
    assert _norm("Make  a FILE!!  named  Foo.Bar") == "make a file named foo bar"


def test_audit_flags_overlap():
    train = [
        _train("please create a config file named app.yaml with port 8080"),
        _train("unrelated task about cats"),
    ]
    bench = [
        {"task_id": "b1", "instruction": "Create a config file named app.yaml with port 8080"},
        {"task_id": "b2", "instruction": "task about cats"},
    ]
    res = audit_leakage(train, bench)
    assert res["n_hits"] == 2
    assert res["hit_rate"] == 1.0


def test_audit_no_false_positive_on_short():
    train = [_train("make a file")]
    bench = [{"task_id": "x", "instruction": "file"}]  # too short to count
    res = audit_leakage(train, bench)
    assert res["n_hits"] == 0


def test_audit_no_leak_distinct():
    train = [_train("rename the log directory to archive")]
    bench = [{"task_id": "y", "instruction": "delete the temp cache folder"}]
    res = audit_leakage(train, bench)
    assert res["n_hits"] == 0
