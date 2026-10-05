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


# ── decontaminate ────────────────────────────────────────────────────────────

def _bench(instr: str, bid: str = "b1") -> dict:
    return {"id": bid, "source": "hermes", "instruction": instr}


def test_decontaminate_drops_leaking_rows():
    from src.audit import decontaminate
    # The bench instruction must appear verbatim inside a training row; the
    # audit is a substring check, so a paraphrase does not count.
    bench = [_bench("install the signal cli and verify")]
    train = [_train("unrelated chatter about files"),
             _train("install the signal cli and verify"),
             _train("more unrelated text")]
    kept, dropped, res = decontaminate(train, bench)
    assert len(kept) == 2
    assert dropped == [1]
    assert res["status"] == "clean"


def test_decontaminate_reaches_fixpoint():
    """audit_leakage reports only the first matching row per bench task.

    Dropping that row can expose another carrying the same text, so one pass
    under-reports: on the real ssd candidate a single pass left 3 of 6 hits.
    """
    from src.audit import decontaminate
    instr = "install the signal cli please"
    bench = [_bench(instr)]
    # Three separate rows all carry the same instruction.
    train = [_train(instr), _train(f"prefix {instr}"), _train(f"suffix {instr}")]
    kept, dropped, res = decontaminate(train, bench)
    assert res["status"] == "clean", "must iterate until no leakage remains"
    assert res["n_hits"] == 0
    assert dropped == [0, 1, 2]
    assert kept == []


def test_decontaminate_noop_when_clean():
    from src.audit import decontaminate
    bench = [_bench("a completely unrelated instruction here")]
    train = [_train("nothing to see"), _train("also nothing")]
    kept, dropped, res = decontaminate(train, bench)
    assert kept == train
    assert dropped == []
    assert res["status"] == "clean"


def test_decontaminate_refuses_when_task_id_unresolvable():
    """Rows keyed by task_id cannot be dropped by position; fail closed."""
    from src.audit import decontaminate
    bench = [_bench("install the signal cli please")]
    train = [{"task_id": "t-1",
              "messages": [{"role": "user", "content": "install the signal cli please"}]}]
    kept, dropped, res = decontaminate(train, bench)
    assert res["status"] == "contaminated", "must not claim clean"
    assert kept == train, "must not silently drop keyed rows"
    assert dropped == []


def test_decontaminate_empty_bench_is_clean():
    from src.audit import decontaminate
    kept, dropped, res = decontaminate([_train("anything")], [])
    assert len(kept) == 1 and dropped == []


def test_decontaminate_terminates_on_pathological_input():
    """max_rounds guard: a bench task matching every row must still stop."""
    from src.audit import decontaminate
    bench = [_bench("shared token phrase")]
    train = [_train("shared token phrase %d" % i) for i in range(40)]
    kept, dropped, res = decontaminate(train, bench, max_rounds=5)
    assert res["status"] in {"clean", "contaminated"}
    assert len(kept) <= len(train)
