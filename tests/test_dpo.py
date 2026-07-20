"""Tests for src.dpo (CPU-safe data prep; GPU train path untested)."""
import json
import os

from src.dpo import load_dpo_pairs, train_dpo

_BASE = "/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b"


def _asst(content, name, args):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": name,
                                      "arguments": json.dumps(args)}}]}


def _pair(good="good", bad="bad"):
    return {
        "prompt": [{"role": "user", "content": "fix foo.py"}],
        "chosen": [_asst("", "write_file", {"path": "foo.py", "content": good})],
        "rejected": [_asst("", "write_file", {"path": "foo.py", "content": bad})],
    }


def test_load_dpo_pairs(tmp_path):
    p = tmp_path / "repairs.dpo.jsonl"
    p.write_text(json.dumps(_pair()) + "\n")
    pairs = load_dpo_pairs(p)
    assert len(pairs) == 1
    pr = pairs[0]
    assert set(pr) == {"prompt", "chosen", "rejected"}
    assert pr["chosen"][0]["tool_calls"][0]["function"]["name"] == "write_file"
    assert json.loads(pr["chosen"][0]["tool_calls"][0]["function"]["arguments"])["content"] == "good"
    assert json.loads(pr["rejected"][0]["tool_calls"][0]["function"]["arguments"])["content"] == "bad"


def test_load_dpo_pairs_empty(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("")
    assert load_dpo_pairs(p) == []


def test_load_dpo_dataset_renders_strings():
    if not os.path.exists(_BASE):
        import pytest
        pytest.skip(f"base model not present: {_BASE}")
    from transformers import AutoTokenizer

    from src.dpo import load_dpo_dataset
    tok = AutoTokenizer.from_pretrained(_BASE)
    ds = load_dpo_dataset(tok, [_pair()] * 3)
    assert len(ds) == 3
    cols = set(ds.column_names)
    assert cols == {"prompt", "chosen", "rejected"}
    for c in cols:
        assert isinstance(ds[0][c], str)
    # chosen carries the corrected call; rejected the bad one
    assert "write_file" in ds[0]["chosen"]
    assert "good" in ds[0]["chosen"]
    assert "bad" in ds[0]["rejected"]


def test_train_dpo_dry_run_builds_dataset():
    # CPU-safe: dry-run loads only the tokenizer (cheap) and builds the
    # (prompt/chosen/rejected) dataset, skipping the GPU model load.
    if not os.path.exists(_BASE):
        import pytest
        pytest.skip(f"base model not present: {_BASE}")
    from src.config import Config
    cfg = Config(raw={})
    pairs = [_pair()] * 3
    rc = train_dpo(cfg, pairs, _BASE, dry_run=True)
    assert rc == 0
