"""Tests for src.eval (tool-call parsing, scoring, held-out split)."""
from __future__ import annotations

import json

from src.eval import parse_tool_calls, score_tool_calls, build_held_out


SEP = "\\u276E\\u276E\\u276E"


def _tc(name, args, body=""):
    return (
        f'<tool_call name="{name}" call_id="c1">\n'
        f'{json.dumps(args)}\n{SEP}\n<tool_result>{body}</tool_result>'
    )


def test_parse_tool_calls_basic():
    text = "pre " + _tc("read", {"filePath": "/a.py", "limit": 10}) + " then " + _tc("bash", {"command": "ls"})
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "read"
    assert calls[0]["args"] == {"filePath": "/a.py", "limit": 10}
    assert calls[1]["name"] == "bash"


def test_parse_tool_calls_literal_sep():
    # The dataset separator is the literal backslash-escaped form, NOT the
    # real U+276E char. Ensure the parser matches the literal text.
    text = _tc("grep", {"pattern": "x"})
    assert len(parse_tool_calls(text)) == 1


def test_parse_tool_calls_malformed_args():
    text = '<tool_call name="bad" call_id="x">\nnot json\n' + SEP + '\n<tool_result></tool_result>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bad"
    assert calls[0]["args"] is None


def test_score_exact_match():
    pred = _tc("read", {"filePath": "/a.py", "limit": 10})
    gold = _tc("read", {"filePath": "/a.py", "limit": 10})
    s = score_tool_calls(pred, gold)
    assert s.total == 1
    assert s.name_correct == 1
    assert s.exact == 1
    assert s.partial == 1


def test_score_partial_match():
    # pred has extra key; gold args are a subset -> partial but not exact
    pred = _tc("read", {"filePath": "/a.py", "limit": 10, "extra": 1})
    gold = _tc("read", {"filePath": "/a.py", "limit": 10})
    s = score_tool_calls(pred, gold)
    assert s.name_correct == 1
    assert s.exact == 0
    assert s.partial == 1


def test_score_wrong_name():
    pred = _tc("grep", {"pattern": "x"})
    gold = _tc("read", {"filePath": "/a.py"})
    s = score_tool_calls(pred, gold)
    assert s.name_correct == 0
    assert s.exact == 0
    assert s.partial == 0


def test_build_held_out(tmp_path):
    d = tmp_path / "datasets"
    d.mkdir()
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(100)]
    (d / "train.ssd.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = build_held_out(str(d), "ssd", frac=0.1, seed=1)
    held = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(held) == 10
    # deterministic: same seed -> same split
    out2 = build_held_out(str(d), "ssd", frac=0.1, seed=1)
    h2 = [json.loads(l) for l in out2.read_text().splitlines() if l.strip()]
    assert [json.dumps(r) for r in held] == [json.dumps(r) for r in h2]


def test_evaluate_all_and_report(monkeypatch, tmp_path):
    """evaluate_all should skip un-trained adapters but still report the base
    baseline, and format_report should render a table. GPU-free via a fake eval."""
    import src.eval as E

    (tmp_path / "toolcall-v5-3b-ssd").mkdir()
    (tmp_path / "toolcall-v5-3b-ssd" / "adapter_config.json").write_text("{}")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "held-out-ssd.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                  {"role": "assistant", "content": "a"}]}) + "\n"
    )
    (eval_dir / "held-out-combined.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q2"},
                                  {"role": "assistant", "content": "a2"}]}) + "\n"
    )

    def fake_evaluate(adapter_path, base_model, held_out_path, rocm=False,
                      max_seq=8192, gen_max_tokens=512, loss_only=False):
        r = E.EvalResult(label="x", adapter=adapter_path)
        r.loss = 1.23 if "ssd" in adapter_path else 2.34
        r.n_held_out = 1
        return r

    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    monkeypatch.setattr(E, "evaluate_baseline",
                        lambda *a, **k: E.EvalResult(label="x", adapter="base", n_held_out=1, loss=3.0))

    results = E.evaluate_all("base-model", str(tmp_path), str(eval_dir), rocm=False)
    # ssd adapter + base(ssd) + base(combined) = 3 rows
    assert len(results) == 3
    adapters = {r.adapter for r in results}
    assert any("ssd" in a for a in adapters)
    assert sum(1 for r in results if r.adapter == "base") == 2

    report = E.format_report(results)
    assert "adapter" in report and "loss" in report


def test_loss_only_skips_generation(monkeypatch, tmp_path):
    """When loss_only=True the model.generate path is not exercised (no GPU
    contention with training)."""
    import src.eval as E
    import torch

    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps({"messages": [{"role": "user", "content": "q"},
                                              {"role": "assistant", "content": "a"}]}) + "\n")

    class StubTok:
        def __call__(self, *a, **k):
            class _T:
                input_ids = torch.zeros((1, 2), dtype=torch.long)
            return _T()
        def apply_chat_template(self, *a, **k):
            return ""

    class StubModel:
        device = "cpu"
        def __call__(self, ids, labels=None):
            class _O:
                loss = torch.tensor(0.5)
            return _O()
        def eval(self):
            pass
        def generate(self, *a, **k):
            raise AssertionError("generate must not be called in loss_only mode")

    monkeypatch.setattr(E, "load_model_and_tokenizer",
                        lambda *a, **k: (StubModel(), StubTok()))
    res = E.evaluate("adapter", "base", str(held), rocm=False, loss_only=True)
    assert res.loss == 0.5
    assert res.tool.total == 0


def test_check_evidence():
    from src.eval import _check_evidence, parse_tool_calls
    # a generated assistant turn that read the expected file and grepped
    gen = ('Let me read /home/scott/git/portfolio-management/run_production.py. '
           '<tool_call name="read" call_id="x"> {"filePath": "/home/scott/git/portfolio-management/run_production.py"} '
           '\\u276E\\u276E\\u276E <tool_result>port=8002</tool_result> The port is 8002.')
    checks = [
        {"kind": "path", "expect": "/home/scott/git/portfolio-management/run_production.py"},
        {"kind": "tool", "expect": "read"},
        {"kind": "substring", "expect": "port"},
    ]
    passed, total, detail = _check_evidence(gen, checks)
    assert total == 3 and passed == 3

    # wrong tool used -> tool check fails
    gen2 = 'I will grep for that. <tool_call name="grep" call_id="y"> {"pattern": "x"} \\u276E\\u276E\\u276E <tool_result></tool_result>'
    checks2 = [{"kind": "tool", "expect": "read"}]
    p2, t2, _ = _check_evidence(gen2, checks2)
    assert t2 == 1 and p2 == 0


def test_load_probe_set(tmp_path):
    from src.eval import load_probe_set
    p = tmp_path / "probe.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}],
                              "checks": [{"kind": "substring", "expect": "q"}]}) + "\n")
    probes = load_probe_set(str(p))
    assert len(probes) == 1
    assert probes[0]["checks"][0]["expect"] == "q"


def test_load_probe_set_label_filter(tmp_path):
    from src.eval import load_probe_set
    p = tmp_path / "probe.jsonl"
    rows = [
        {"label": "ssd", "messages": [{"role": "user", "content": "a"}],
         "checks": [{"kind": "substring", "expect": "a"}]},
        {"label": "nas5-main", "messages": [{"role": "user", "content": "b"}],
         "checks": [{"kind": "substring", "expect": "b"}]},
        {"messages": [{"role": "user", "content": "c"}],
         "checks": [{"kind": "substring", "expect": "c"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # no label -> all 3
    assert len(load_probe_set(str(p))) == 3
    # scoped -> only that label's rows + the unlabeled generic one
    scoped = load_probe_set(str(p), label="ssd")
    assert [r["messages"][0]["content"] for r in scoped] == ["a", "c"]
    nas = load_probe_set(str(p), label="nas5-main")
    assert [r["messages"][0]["content"] for r in nas] == ["b", "c"]


def test_best_adapter_and_report(tmp_path):
    from src.eval import EvalResult, ToolCallScore, write_report, best_adapter
    base = EvalResult(label="base", adapter="base", loss=3.0, n_held_out=5)
    a1 = EvalResult(label="ssd", adapter="ssd", loss=1.2, n_held_out=5)
    a2 = EvalResult(label="nas5", adapter="nas5", loss=0.9, n_held_out=5)
    t = ToolCallScore(); t.total = 1; t.exact = 1
    a2.tool = t
    results = [base, a1, a2]

    assert best_adapter(results, "loss").adapter == "nas5"
    assert best_adapter(results, "tool_exact").adapter == "nas5"
    assert best_adapter([base], "loss") is None

    path = write_report(results, out_dir=str(tmp_path), tag="t")
    assert path.endswith(".md")
    import os
    assert os.path.exists(path)
    assert os.path.exists(path.replace(".md", ".json"))


def test_sanity_check_adapters(monkeypatch, tmp_path):
    from src.eval import sanity_check_adapters
    import src.eval as E
    (tmp_path / "toolcall-v5-3b-ssd").mkdir()
    (tmp_path / "toolcall-v5-3b-ssd" / "adapter_config.json").write_text("{}")
    # no adapter_config for nas5 -> "missing"

    import torch
    class StubTok:
        def __call__(self, *a, **k):
            class _T:
                input_ids = torch.zeros((1, 1), dtype=torch.long)
            return _T()
    class StubOut:
        logits = object()
    class StubModel:
        device = "cpu"
        def __call__(self, ids, labels=None):
            return StubOut()
        def eval(self):
            pass

    monkeypatch.setattr(E, "load_model_and_tokenizer",
                        lambda *a, **k: (StubModel(), StubTok()))
    status = sanity_check_adapters("base", str(tmp_path), ["ssd", "nas5"], rocm=False)
    assert status["ssd"] == "ok"
    assert status["nas5"] == "missing"


def test_merge_adapter(monkeypatch, tmp_path):
    """merge_adapter should fuse the LoRA into base and save a standalone model.
    GPU-free via stubbed transformers/peft."""
    import torch
    import src.merge as M

    (tmp_path / "toolcall-v5-3b-ssd").mkdir()
    (tmp_path / "toolcall-v5-3b-ssd" / "adapter_config.json").write_text("{}")

    class StubTok:
        def save_pretrained(self, *a, **k):
            pass
    class StubMerged:
        def save_pretrained(self, *a, **k):
            pass
    class StubPeft:
        def merge_and_unload(self):
            return StubMerged()
    class StubBase:
        pass

    import sys
    import peft as _peft
    _tf = sys.modules["transformers"]
    monkeypatch.setattr(_tf, "AutoModelForCausalLM",
                        type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: StubBase())}))
    monkeypatch.setattr(_tf, "AutoTokenizer",
                        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: StubTok())}))
    monkeypatch.setattr(_peft, "PeftModel",
                        type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: StubPeft())}))

    out = tmp_path / "merged"
    res = M.merge_adapter(str(tmp_path / "toolcall-v5-3b-ssd"), "base", str(out), rocm=False)
    assert res == str(out)
    assert out.exists()
