"""Quantize preflight + LFM3 template regressions (synthetic telemetry only).

No GPU, no model download and no network: the memory/inventory gates and the
heavy model imports are replaced with deterministic stand-ins.
"""
from __future__ import annotations

import json
import os
import sys
import types
from importlib.machinery import ModuleSpec

import pytest
import transformers  # noqa: F401  -- resolve lazy imports before sys.modules stubs
from conftest import make_cfg
from transformers import AutoTokenizer  # noqa: F401

from src import lfm3_support as support
from src import quantize as Q
from src import training_admission as A

GIB = 1024 * 1024
MIN = Q.MIN_QUANTIZE_AVAILABLE_KIB
IDLE_INVENTORY = [{"type": "embedding", "loaded_instances": []}]
RESIDENT_LLM = IDLE_INVENTORY + [
    {"type": "llm", "loaded_instances": [{"id": "ornith"}]},
]


class _Response:
    status = 200

    def __init__(self, models):
        self._models = models

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size):
        return json.dumps({"models": self._models}).encode()


class _Opener:
    def __init__(self, models):
        self._models = models

    def open(self, _url, timeout=3):
        return _Response(self._models)


@pytest.fixture
def telemetry(monkeypatch, tmp_path):
    """Synthetic xwing meminfo + inventory; returns (set_available, set_inventory)."""
    meminfo = tmp_path / "meminfo"
    state = {"models": IDLE_INVENTORY}

    def set_available(kib):
        meminfo.write_text(f"MemTotal: 32 GiB\nMemAvailable: {kib} kB\n")

    def set_inventory(models):
        state["models"] = models

    set_available(MIN + 128 * GIB)
    monkeypatch.setattr(A, "MEMINFO", meminfo)
    monkeypatch.setattr(A.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _Opener(state["models"]))
    return set_available, set_inventory


@pytest.fixture
def forbid_model_load(monkeypatch):
    """Any heavy import/model load during a refusal is an explicit failure."""
    def forbidden(*a, **k):
        raise AssertionError("model load started before preflight refusal")

    for modname, attrs in (
        ("torch", {"bfloat16": "bf16"}),
        ("peft", {"PeftModel": types.SimpleNamespace(from_pretrained=forbidden)}),
        ("transformers", {
            "AutoModelForCausalLM": types.SimpleNamespace(from_pretrained=forbidden),
            "AutoTokenizer": types.SimpleNamespace(from_pretrained=forbidden),
        }),
        ("auto_gptq", {
            "AutoGPTQForCausalLM": types.SimpleNamespace(from_pretrained=forbidden),
            "BaseQuantizeConfig": object,
        }),
        ("awq", {"AutoAWQForCausalLM": types.SimpleNamespace(from_pretrained=forbidden)}),
    ):
        mod = types.ModuleType(modname)
        mod.__spec__ = ModuleSpec(modname, None)
        for name, value in attrs.items():
            setattr(mod, name, value)
        monkeypatch.setitem(sys.modules, modname, mod)


class _DummyTok:
    chat_template = None
    pad_token_id = 1

    def __call__(self, text, return_tensors=None):
        class Enc:
            input_ids = [1, 2, 3]
        return Enc()

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.txt"), "w") as fh:
            fh.write("tok")


class _DummyModel:
    def quantize(self, *args, **kwargs):
        self.cal_data = list(args)

    def save_quantized(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "q.bin"), "wb") as fh:
            fh.write(b"x" * 1024)


def _install_fake_backend(monkeypatch, backend="gptq", model=None):
    """Register a usable auto_gptq/autoawq stand-in."""
    if backend == "gptq":
        mod = types.ModuleType("auto_gptq")
        mod.AutoGPTQForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: model)
        mod.BaseQuantizeConfig = type("BaseQuantizeConfig", (), {
            "__init__": lambda self, **k: None})
    else:
        mod = types.ModuleType("awq")
        mod.AutoAWQForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: model)
    mod.__spec__ = ModuleSpec(mod.__name__, None)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _DummyTok()))


def _merged(tmp_path):
    src = tmp_path / "toolcall-v5-3b-LFM3-merged"
    src.mkdir(exist_ok=True)
    (src / "config.json").write_text("{}")
    return str(src)


# --- preflight gates --------------------------------------------------------

def test_quantize_memory_floor_matches_training_admission():
    assert Q.MIN_QUANTIZE_AVAILABLE_KIB == A.MIN_KIB


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_low_memory_refuses_before_any_load(telemetry, forbid_model_load, tmp_path, method):
    set_available, _ = telemetry
    set_available(MIN - 1)
    src, out = _merged(tmp_path), str(tmp_path / "out")
    fn = Q.quantize_gptq if method == "gptq" else Q.quantize_awq
    with pytest.raises(Q.QuantizePreflightError, match="MemAvailable"):
        fn(src, out)
    assert not os.path.exists(out)


def test_unreadable_meminfo_fails_closed(telemetry, forbid_model_load, tmp_path):
    A.MEMINFO.unlink()
    with pytest.raises(Q.QuantizePreflightError, match="cannot read MemAvailable"):
        Q.quantize_gptq(_merged(tmp_path), str(tmp_path / "out"))


@pytest.mark.parametrize("models", [
    None,
    [],
    [{"type": "llm"}],
    [{"type": "embedding", "loaded_instances": "loaded"}],
])
def test_unusable_inventory_fails_closed(telemetry, forbid_model_load, tmp_path, models):
    _, set_inventory = telemetry
    set_inventory(models)
    with pytest.raises(Q.QuantizePreflightError,
                       match=r"model_inventory_(unavailable|invalid)"):
        Q.quantize_gptq(_merged(tmp_path), str(tmp_path / "out"))


def test_unavailable_inventory_fails_closed(telemetry, forbid_model_load, tmp_path,
                                            monkeypatch):
    def down(*a, **k):
        raise OSError("inventory endpoint down")
    monkeypatch.setattr(A, "build_opener", down)
    with pytest.raises(Q.QuantizePreflightError, match="inventory unavailable"):
        Q.quantize_gptq(_merged(tmp_path), str(tmp_path / "out"))


def test_resident_llm_refuses_despite_headroom(telemetry, forbid_model_load, tmp_path):
    _, set_inventory = telemetry
    set_inventory(RESIDENT_LLM)
    with pytest.raises(Q.QuantizePreflightError, match="resident LLM"):
        Q.quantize_gptq(_merged(tmp_path), str(tmp_path / "out"))


def test_other_host_skips_inventory_but_keeps_memory_guard(telemetry, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("inventory consulted on non-xwing host")
    set_available, _ = telemetry
    monkeypatch.setattr(A.socket, "gethostname", lambda: "other-box")
    monkeypatch.setattr(A, "build_opener", forbidden)
    set_available(MIN - 1)
    with pytest.raises(Q.QuantizePreflightError, match="MemAvailable"):
        Q.preflight_quantize()


def test_idle_inventory_admits(telemetry):
    assert Q.preflight_quantize() is None


def test_cli_refuses_with_78_before_output(telemetry, monkeypatch, tmp_path, capsys):
    set_available, _ = telemetry
    set_available(2 * GIB)
    out_base = tmp_path / "checkpoints"
    src = out_base / "toolcall-v5-3b-x-merged"
    src.mkdir(parents=True)
    cfg = make_cfg(train={"output_dir": str(out_base)})
    assert Q.main(cfg, ["cli", "quantize", "--label=x"]) == 78
    printed = capsys.readouterr().out
    assert "[error]" in printed and "MemAvailable" in printed
    assert not (out_base / "toolcall-v5-3b-x-merged-4bit").exists()


# --- LFM3 chat-template propagation ----------------------------------------

def _recording_loader(calls):
    def fake(tok, name, *extra):
        calls.append((tok, name, extra))
        return True
    return fake


def test_gptq_uses_shared_fail_closed_template_loader(telemetry, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(support, "load_lfm3_chat_template", _recording_loader(calls))
    monkeypatch.setattr(support, "patch_transformers_for_lfm3", lambda: None)
    model = _DummyModel()
    _install_fake_backend(monkeypatch, "gptq", model)
    src = _merged(tmp_path)

    result = Q.quantize_gptq(src, str(tmp_path / "out"), calibration_texts=["calibrate"])
    assert result.success, result.message
    assert len(calls) == 1
    assert calls[0][1] == src  # shared loader receives the merged LFM3 path
    assert calls[0][0].__class__ is _DummyTok


def test_awq_uses_shared_fail_closed_template_loader(telemetry, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(support, "load_lfm3_chat_template", _recording_loader(calls))
    monkeypatch.setattr(support, "patch_transformers_for_lfm3", lambda: None)
    _install_fake_backend(monkeypatch, "awq", _DummyModel())
    src = _merged(tmp_path)

    result = Q.quantize_awq(src, str(tmp_path / "out"))
    assert result.success, result.message
    assert len(calls) == 1
    assert calls[0][1] == src


def test_missing_lfm3_template_fails_closed(telemetry, monkeypatch, tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError("LFM3 chat_template.jinja not found")
    monkeypatch.setattr(support, "load_lfm3_chat_template", missing)
    monkeypatch.setattr(support, "patch_transformers_for_lfm3", lambda: None)
    _install_fake_backend(monkeypatch, "gptq", _DummyModel())

    out = str(tmp_path / "out")
    result = Q.quantize_gptq(_merged(tmp_path), out, calibration_texts=["calibrate"])
    assert result.success is False
    assert "chat_template.jinja" in result.message
    assert not os.path.exists(out)
