"""Coverage tests for pure-logic helpers in eval.py (no GPU needed)."""
from __future__ import annotations

from src import eval as EV

SEP = "\\u276E\\u276E\\u276E"


def _call(name, args):
    return f'<tool_call name="{name}" call_id="c1">{args}{SEP}'


def test_parse_tool_calls_valid():
    text = _call("read", '{"filePath": "a"}')
    out = EV.parse_tool_calls(text)
    assert out == [{"name": "read", "args": {"filePath": "a"}}]


def test_parse_tool_calls_malformed_args():
    text = _call("read", "not json")
    out = EV.parse_tool_calls(text)
    assert out[0]["name"] == "read"
    assert out[0]["args"] is None


def test_parse_tool_calls_empty():
    assert EV.parse_tool_calls("no calls here") == []


def test_args_match_gold_none():
    assert EV._args_match({"a": 1}, None) is True


def test_args_match_subset():
    assert EV._args_match({"a": 1}, {"a": 1}) is True
    assert EV._args_match({"a": 1}, {"a": 1, "b": 2}) is False


def test_toolcallscore_properties_zero():
    s = EV.ToolCallScore()
    assert s.name_accuracy == 0.0
    assert s.partial_match == 0.0
    assert s.exact_match == 0.0


def test_score_tool_calls_exact():
    gold = _call("read", '{"filePath": "a"}')
    s = EV.score_tool_calls(gold, gold)
    assert s.name_correct == 1
    assert s.exact == 1
    assert s.partial == 1


def test_score_tool_calls_partial_args():
    gold = _call("read", '{"filePath": "a"}')
    pred = _call("read", '{"filePath": "a", "extra": 1}')
    s = EV.score_tool_calls(pred, gold)
    assert s.name_correct == 1
    assert s.exact == 0
    assert s.partial == 1


def test_score_tool_calls_wrong_name():
    gold = _call("read", '{"filePath": "a"}')
    pred = _call("write", '{"filePath": "a"}')
    s = EV.score_tool_calls(pred, gold)
    assert s.name_correct == 0


def test_best_adapter_loss():
    a = EV.EvalResult(label="x", adapter="m/x", loss=1.2, tool=EV.ToolCallScore())
    b = EV.EvalResult(label="y", adapter="m/y", loss=0.8, tool=EV.ToolCallScore())
    base = EV.EvalResult(label="b", adapter="base", loss=5.0, tool=EV.ToolCallScore())
    best = EV.best_adapter([a, b, base], metric="loss")
    assert best.adapter == "m/y"


def test_best_adapter_tool_exact():
    a = EV.EvalResult(label="x", adapter="m/x", tool=EV.ToolCallScore(exact=1, total=2))
    b = EV.EvalResult(label="y", adapter="m/y", tool=EV.ToolCallScore(exact=3, total=3))
    best = EV.best_adapter([a, b], metric="tool_exact")
    assert best.adapter == "m/y"


def test_best_adapter_no_adapters():
    base = EV.EvalResult(label="b", adapter="base", loss=1.0, tool=EV.ToolCallScore())
    assert EV.best_adapter([base]) is None


def test_format_report_contains_header():
    a = EV.EvalResult(label="x", adapter="ax", n_held_out=10,
                      loss=1.0, perplexity=2.7, tool=EV.ToolCallScore(total=1, name_correct=1))
    out = EV.format_report([a])
    assert "adapter" in out
    assert "ax" in out


def test_check_evidence_tool_and_substring():
    gen = _call("bash", '{}') + "hello world"
    passed, total, detail = EV._check_evidence(
        gen, [{"kind": "tool", "expect": "bash"}, {"kind": "substring", "expect": "hello"}])
    assert total == 2
    assert passed == 2


def test_probe_result_check_rate():
    p = EV.ProbeResult(adapter="x", checks_passed=3, checks_total=4)
    assert abs(p.check_rate - 0.75) < 1e-9
    assert EV.ProbeResult(adapter="y").check_rate == 0.0


def test_format_probe_comparison():
    p = EV.ProbeResult(adapter="base", n_probes=2, checks_passed=1, checks_total=2)
    out = EV.format_probe_comparison([p])
    assert "check rate" in out
    assert "base" in out


def test_load_probe_set_label_scoping(tmp_path):
    p = tmp_path / "probe.jsonl"
    p.write_text('\n'.join([
        '{"label": "ssd", "messages": [], "checks": []}',
        '{"messages": [], "checks": []}',
        '{"label": "other", "messages": [], "checks": []}',
    ]) + "\n")
    rows = EV.load_probe_set(str(p), label="ssd")
    assert len(rows) == 2  # one scoped + one generic
    assert all(r.get("label") in ("ssd", None) for r in rows)
