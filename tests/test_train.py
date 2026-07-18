"""Tests for src.train (backend selection, dataset loading, dry-run)."""
from __future__ import annotations

import json

import pytest

from src.config import Config, _DEFAULTS
from conftest import make_cfg
import copy
from src.train import (_detect_cuda, _detect_rocm, _resolve_backend, _to_messages,
                       load_dataset, validate_dataset, TrainError, _render_sample_texts)


def _cfg(**train_overrides):
    raw = copy.deepcopy(_DEFAULTS)
    raw["train"].update(train_overrides)
    return Config(raw=raw)


def test_detect_cuda_no_torch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert _detect_cuda() is False


def test_detect_rocm_env(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    assert _detect_rocm() is True
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setenv("HSA_PATH", "/opt/hsa")
    assert _detect_rocm() is True
    monkeypatch.delenv("HSA_PATH", raising=False)


def test_detect_rocm_injected():
    assert _detect_rocm(rocm_path="/x") is True
    assert _detect_rocm(rocm_path=None) is False
    assert _detect_rocm(rocm_path="") is False


def test_resolve_backend_explicit():
    assert _resolve_backend(_cfg(backend="unsloth")) == "unsloth"
    assert _resolve_backend(_cfg(backend="peft")) == "peft"


def test_resolve_backend_auto_no_cuda(monkeypatch):
    monkeypatch.setattr("src.train._detect_cuda", lambda: False)
    assert _resolve_backend(_cfg(backend="auto")) == "peft"


def test_resolve_backend_auto_cuda_no_unsloth(monkeypatch):
    monkeypatch.setattr("src.train._detect_cuda", lambda: True)

    def _import(name, *a, **k):
        if name == "unsloth":
            raise ImportError("no unsloth")
        return __import__(name, *a, **k)
    monkeypatch.setattr("builtins.__import__", _import)
    assert _resolve_backend(_cfg(backend="auto")) == "peft"


def test_resolve_backend_auto_cuda_with_unsloth(monkeypatch):
    monkeypatch.setattr("src.train._detect_cuda", lambda: True)
    import sys
    sys.modules.setdefault("unsloth", type("U", (), {})())
    try:
        assert _resolve_backend(_cfg(backend="auto")) == "unsloth"
    finally:
        sys.modules.pop("unsloth", None)


def test_load_dataset(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"b": 2}) + "\n\n")
    rows = load_dataset(str(p))
    assert rows == [{"a": 1}, {"b": 2}]


def test_load_dataset_malformed_raises(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"a":1}\nNOT JSON\n')
    with pytest.raises(TrainError):
        load_dataset(str(p))


def test_validate_dataset_missing(tmp_path):
    with pytest.raises(TrainError):
        validate_dataset(str(tmp_path / "nope.jsonl"))


def test_validate_dataset_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(TrainError):
        validate_dataset(str(p))


def test_validate_dataset_ok(tmp_path):
    p = tmp_path / "ok.jsonl"
    p.write_text(json.dumps({"x": 1}) + "\n")
    assert validate_dataset(str(p)) == [{"x": 1}]


def test_to_messages_chatml():
    ex = {"messages": [{"role": "user", "content": "q"}]}
    assert _to_messages(ex) == [{"role": "user", "content": "q"}]


def test_to_messages_sharegpt():
    ex = {"conversations": [{"from": "human", "value": "q"},
                            {"from": "gpt", "value": "a"},
                            {"from": "system", "value": "s"}]}
    msgs = _to_messages(ex)
    assert {"role": "system", "content": "s"} in msgs
    assert {"role": "user", "content": "q"} in msgs
    assert {"role": "assistant", "content": "a"} in msgs


def test_to_messages_alpaca():
    ex = {"instruction": "i", "output": "o"}
    msgs = _to_messages(ex)
    assert msgs == [{"role": "user", "content": "i"},
                    {"role": "assistant", "content": "o"}]


def test_render_sample_texts_with_fake_tokenizer():
    class FakeTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "|".join(f"{m['role']}:{m['content']}" for m in msgs)
    data = [{"messages": [{"role": "user", "content": "q"}]}]
    out = _render_sample_texts(data, FakeTok(), 8192, n=3)
    assert out == ["user:q"]


def test_dry_run_writes_nothing_but_validates(tmp_path, capsys):
    # Build a minimal dataset and run dry-run with a fake tokenizer import.
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "train.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
    raw = copy.deepcopy(_DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(ds)}
    raw["train"]["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
    cfg = Config(raw=raw)

    import src.train as T
    real = T.AutoTokenizer if hasattr(T, "AutoTokenizer") else None

    class FakeTok:
        @staticmethod
        def from_pretrained(name):
            return FakeTok()
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "rendered"
    monkeypatch_tok = FakeTok

    # Patch transformers import used inside dry-run
    import sys, types
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = FakeTok
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_tf
    try:
        rc = T.main(cfg, dry_run=True)
    finally:
        if saved:
            sys.modules["transformers"] = saved
        else:
            sys.modules.pop("transformers", None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
