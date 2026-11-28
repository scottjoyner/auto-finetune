"""All offline stages route LFM3 chat templates through the shared loader.

The loader in ``src.lfm3_support`` fails closed when chat_template.jinja is
absent. These regressions pin that binarize, dpo, bench, export, pretokenize
and quantize all call it instead of growing a divergent inline fallback.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from conftest import make_cfg
from transformers import AutoTokenizer

import src
from src import lfm3_support as support
from src import model_export as ME
from src import pretokenize

LFM3 = "LiquidAI/LFM3-1.7B"
SRC_DIR = Path(src.__file__).parent
PROPAGATED_MODULES = (
    "binarize",
    "dpo",
    "bench",
    "model_export",
    "pretokenize",
    "quantize",
)


class _DummyTok:
    chat_template = None
    pad_token_id = 1
    eos_token = "</s>"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.txt"), "w") as fh:
            fh.write("tok")


@pytest.fixture
def loader_calls(monkeypatch):
    """Record every call to the shared loader; neutralise the RoPE patches."""
    calls = []

    def fake(tokenizer, model_name, *extra):
        calls.append((tokenizer, model_name, extra))
        return True

    monkeypatch.setattr(support, "load_lfm3_chat_template", fake)
    monkeypatch.setattr(support, "patch_transformers_for_lfm3", lambda: None)
    return calls


def _assert_loader_used(calls, model=LFM3):
    assert len(calls) == 1, calls
    assert calls[0][1] == model
    assert isinstance(calls[0][0], _DummyTok)


# --- every stage references the shared loader, none defines its own ---------

@pytest.mark.parametrize("name", PROPAGATED_MODULES)
def test_stage_uses_shared_loader_without_inline_fallback(name):
    text = (SRC_DIR / f"{name}.py").read_text()
    assert "load_lfm3_chat_template" in text
    # No stage may read the template file itself or overwrite a template
    # inline; the shared loader owns both behaviours.
    assert "chat_template.jinja" not in text
    assert "chat_template =" not in text


# --- behavioural call-site checks -------------------------------------------

def test_pretokenize_loads_template(loader_calls, monkeypatch):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))
    tok = pretokenize._load_tokenizer(LFM3)
    assert isinstance(tok, _DummyTok)
    _assert_loader_used(loader_calls)


def test_binarize_loads_template(loader_calls, monkeypatch, tmp_path):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))
    src = tmp_path / "mix.jsonl"
    src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    from src.binarize import binarize_mix
    assert binarize_mix(str(src), str(tmp_path / "arrow"), LFM3, 128) == 1
    _assert_loader_used(loader_calls)


def test_dpo_loads_template(loader_calls, monkeypatch):
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **k: None
    peft.get_peft_model = lambda model, cfg: model
    trl = types.ModuleType("trl")
    trl.DPOTrainer = object
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))

    from src.dpo import train_dpo
    pair = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    assert train_dpo(make_cfg(), [pair], LFM3, dry_run=True) == 0
    _assert_loader_used(loader_calls)


def test_bench_local_driver_loads_template(loader_calls, monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))

    def fake_model(*a, **k):
        return types.SimpleNamespace(device="cpu")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained",
                        staticmethod(fake_model))
    from src.bench import LocalDriver
    driver = LocalDriver(LFM3)
    driver._load()
    assert isinstance(driver._tok, _DummyTok)
    _assert_loader_used(loader_calls)


@pytest.mark.parametrize("export_fn", ["export_merged", "export_onnx"])
def test_export_loads_template(loader_calls, monkeypatch, tmp_path, export_fn):
    # model_export imports the helpers at module load time, so patch its own
    # bindings as well as the support module's.
    def record(tok, name, *extra):
        loader_calls.append((tok, name, extra))
        return True

    monkeypatch.setattr(ME, "load_lfm3_chat_template", record)
    monkeypatch.setattr(ME, "patch_transformers_for_lfm3", lambda: None)

    def fake_model(*a, **k):
        return types.SimpleNamespace(save_pretrained=lambda path: None)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.onnx = types.SimpleNamespace(export=lambda *a, **k: None)
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda *a, **k:
        types.SimpleNamespace(merge_and_unload=lambda: fake_model()))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained",
                        staticmethod(fake_model))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))

    result = getattr(ME, export_fn)("ckpt", str(tmp_path / "out"), base_model=LFM3)
    assert result.success, result.error
    _assert_loader_used(loader_calls)
