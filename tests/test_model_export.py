"""Offline coverage for ``src.model_export`` happy paths.

Companion to ``test_model_export_pure.py``: that module pins the pure and
control-flow surface and deliberately refuses to let any model load run. This
module exercises the *mocked* load/merge/export paths so the LFM3-dependent
decisions are pinned without downloading weights or touching a GPU.

Heavy imports (torch / transformers / peft) are replaced with in-module fakes
via ``sys.modules``. In particular CI does not install ``peft``, so patching the
real package would error there; the fakes keep this suite hermetic.
"""
from __future__ import annotations

import inspect
import os
import sys
import types

import pytest

from src import model_export as ME
from src.model_export import ExportResult, export_gguf, export_merged, export_onnx, main


# --- hermetic heavy stack ----------------------------------------------------

@pytest.fixture
def fake_stack(monkeypatch):
    """Stub torch/transformers/peft and record loader kwargs."""
    calls = {"model": [], "tokenizer": [], "onnx_export": []}

    class _StubMerged:
        def save_pretrained(self, path):
            calls.setdefault("merged_saved", []).append(path)

    class _StubPeft:
        def merge_and_unload(self):
            return _StubMerged()

    class _StubTok:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

        def save_pretrained(self, path):
            calls.setdefault("tok_saved", []).append(path)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.onnx = types.SimpleNamespace(
        export=lambda *a, **k: calls["onnx_export"].append((a, k)))

    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubPeft())

    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: (calls["model"].append(dict(k)) or object()))
    tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: (calls["tokenizer"].append(dict(k)) or _StubTok()))

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    return calls


# --- import surface / dataclass / signatures --------------------------------

def test_import_model_export():
    assert ME.ExportResult is ExportResult
    assert callable(export_merged)
    assert callable(export_gguf)
    assert callable(export_onnx)
    assert callable(main)


def test_export_result_defaults():
    r = ExportResult(format="merged", output_path="/tmp/out")
    assert (r.format, r.output_path, r.size_bytes, r.success, r.error,
            r.duration_seconds) == ("merged", "/tmp/out", 0, False, None, 0.0)


def test_export_result_fields():
    r = ExportResult("onnx", "/tmp/o", size_bytes=12, success=True,
                     error=None, duration_seconds=1.5)
    assert (r.success, r.size_bytes, r.duration_seconds) == (True, 12, 1.5)


def test_export_signatures():
    assert list(inspect.signature(export_merged).parameters) == [
        "checkpoint_path", "output_path", "base_model"]
    assert list(inspect.signature(export_gguf).parameters) == [
        "checkpoint_path", "output_path", "quantize"]
    assert list(inspect.signature(export_onnx).parameters) == [
        "checkpoint_path", "output_path", "base_model"]
    assert list(inspect.signature(main).parameters) == ["argv"]


# --- early failure -----------------------------------------------------------

def test_export_gguf_no_convert_script(tmp_path):
    r = export_gguf(str(tmp_path / "ckpt"), str(tmp_path / "out.gguf"))
    assert r.success is False
    assert r.error == "llama.cpp convert script not found"


def test_main_missing_required_args():
    with pytest.raises(SystemExit):
        main([])


# --- mocked happy paths: the intended `_lfm3` decision ----------------------

def test_export_merged_non_lfm3_does_not_trust_remote_code(fake_stack, tmp_path):
    r = export_merged(str(tmp_path / "ckpt"), str(tmp_path / "merged"))
    assert r.success is True
    assert r.error is None
    assert r.format == "merged"
    assert fake_stack["model"][-1]["trust_remote_code"] is False
    assert fake_stack["tokenizer"][-1]["trust_remote_code"] is False


def test_export_merged_lfm3_trusts_remote_code(fake_stack, tmp_path, monkeypatch):
    patched, templates = [], []
    monkeypatch.setattr(ME, "is_lfm3_model", lambda name: True)
    monkeypatch.setattr(ME, "patch_transformers_for_lfm3", lambda: patched.append(True))
    monkeypatch.setattr(ME, "load_lfm3_chat_template",
                        lambda tok, name: templates.append(name))

    r = export_merged(str(tmp_path / "ckpt"), str(tmp_path / "merged"),
                      "local/LFM3-1.7B")

    assert r.success is True
    assert fake_stack["model"][-1]["trust_remote_code"] is True
    assert fake_stack["tokenizer"][-1]["trust_remote_code"] is True
    assert patched == [True]
    assert templates == ["local/LFM3-1.7B"]


def test_export_onnx_non_lfm3_does_not_trust_remote_code(fake_stack, tmp_path):
    r = export_onnx(str(tmp_path / "ckpt"), str(tmp_path / "onnx"))
    assert r.success is True
    assert r.format == "onnx"
    assert fake_stack["model"][-1]["trust_remote_code"] is False
    assert fake_stack["tokenizer"][-1]["trust_remote_code"] is False
    assert len(fake_stack["onnx_export"]) == 1


def test_export_onnx_lfm3_trusts_remote_code(fake_stack, tmp_path, monkeypatch):
    monkeypatch.setattr(ME, "is_lfm3_model", lambda name: True)
    monkeypatch.setattr(ME, "patch_transformers_for_lfm3", lambda: None)
    monkeypatch.setattr(ME, "load_lfm3_chat_template", lambda *a: None)

    r = export_onnx(str(tmp_path / "ckpt"), str(tmp_path / "onnx"),
                    "local/LFM3-1.7B")

    assert r.success is True
    assert fake_stack["model"][-1]["trust_remote_code"] is True
    assert fake_stack["tokenizer"][-1]["trust_remote_code"] is True


def test_main_merged_mocked(fake_stack, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    rc = main(["--checkpoint", ckpt, "--output", str(tmp_path / "merged"),
               "--format", "merged"])
    assert rc == 0
    assert fake_stack["model"][-1]["trust_remote_code"] is False
