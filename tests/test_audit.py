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


def test_audit_supports_bench_id_prompt_schema_and_unique_hit_rate():
    train = [_train("create alpha configuration with port 9000"),
             _train("create beta configuration with port 9001")]
    bench = [
        {"id": "a", "prompt": "create alpha configuration with port 9000"},
        {"id": "b", "prompt": "create beta configuration with port 9001"},
        {"id": "c", "prompt": "a genuinely unrelated benchmark instruction"},
    ]
    res = audit_leakage(train, bench)
    assert [h["bench_task_id"] for h in res["hits"]] == ["a", "b"]
    assert [h["instruction"] for h in res["hits"]] == [bench[0]["prompt"], bench[1]["prompt"]]
    assert res["n_hits"] == 2
    assert res["hit_rate"] == 2 / 3
    assert res["status"] == "contaminated"


def test_audit_detects_chat_format_holdout_copied_from_train():
    row = {"messages": [
        {"role": "user", "content": "implement the durable queue worker"},
        {"role": "assistant", "content": "I will add tests first."},
    ]}
    res = audit_leakage([row], [row])
    assert res["n_eligible"] == 1
    assert res["n_hits"] == 1
    assert res["status"] == "contaminated"


def test_audit_fails_closed_when_nonempty_benchmark_has_no_evaluable_text():
    res = audit_leakage([], [{"metadata": {"name": "opaque"}}])
    assert res["n_eligible"] == 0
    assert res["status"] == "not_evaluable"
