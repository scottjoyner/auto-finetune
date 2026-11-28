"""Merge preflight regressions: synthetic /proc data, no model load or network."""
from __future__ import annotations

import json
import sys
import types

import pytest

from src import merge as M
from src import training_admission as A

GIB = 1024 * 1024


class _FakeResponse:
    status = 200

    def __init__(self, models):
        self._models = models

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size):
        return json.dumps({"models": self._models}).encode()


class _FakeOpener:
    def __init__(self, models):
        self._models = models

    def open(self, _url, timeout=3):
        return _FakeResponse(self._models)


@pytest.fixture
def host(monkeypatch, tmp_path):
    """Synthetic MemAvailable + xwing identity; returns a memory setter."""
    meminfo = tmp_path / "meminfo"

    def set_available(kib):
        meminfo.write_text(f"MemTotal: 16 GiB\nMemAvailable: {kib} kB\n")

    set_available(32 * GIB)
    monkeypatch.setattr(A, "MEMINFO", meminfo)
    monkeypatch.setattr(A.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _FakeOpener([]))
    return set_available


@pytest.fixture
def forbid_model_load(monkeypatch):
    """Any torch/peft/transformers model load is a test failure."""
    # Import the cli/train graph while the real transformers package is still
    # in sys.modules: src.train eagerly imports transformers.configuration_utils
    # and would treat a stubbed top-level module as a non-package.
    from src import cli  # noqa: F401

    def forbidden(*a, **k):
        raise AssertionError("model load started before preflight refusal")

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float16 = "fp16"
    torch.version = types.SimpleNamespace(hip=None)
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=forbidden)
    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=forbidden)
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=forbidden)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", tf)


def _merge():
    M.merge_adapter("adapter", "base", "/tmp/never-created")


def test_merge_floor_covers_training_memory_gate():
    """The merge floor must not be looser than the 16 GiB training gate."""
    assert M.MIN_MERGE_AVAILABLE_KIB >= A.MIN_KIB


def test_low_meminfo_refuses_before_any_load(host, forbid_model_load):
    host(2 * GIB)
    with pytest.raises(M.MergePreflightError, match="MemAvailable"):
        _merge()


def test_missing_meminfo_entry_fails_closed(host, forbid_model_load):
    host(32 * GIB)
    A.MEMINFO.write_text("MemTotal: 16 GiB\n")
    with pytest.raises(M.MergePreflightError, match="MemAvailable"):
        _merge()


def test_malformed_meminfo_fails_closed(host, forbid_model_load):
    host(32 * GIB)
    A.MEMINFO.write_text("MemAvailable: plenty\n")
    with pytest.raises(M.MergePreflightError, match="cannot read MemAvailable"):
        _merge()


def test_resident_llm_refuses_despite_headroom(host, forbid_model_load, monkeypatch):
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _FakeOpener([
        {"type": "embedding", "loaded_instances": [{"id": "embed"}]},
        {"type": "llm", "loaded_instances": [{"id": "ornith"}]},
    ]))
    with pytest.raises(M.MergePreflightError, match="resident LLM"):
        _merge()


def test_unavailable_inventory_fails_closed_on_xwing(host, forbid_model_load, monkeypatch):
    def unavailable(*a, **k):
        raise OSError("inventory endpoint down")

    monkeypatch.setattr(A, "build_opener", unavailable)
    with pytest.raises(M.MergePreflightError, match="inventory unavailable"):
        _merge()


def test_invalid_inventory_fails_closed(host, forbid_model_load, monkeypatch):
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _FakeOpener(None))
    with pytest.raises(M.MergePreflightError, match="model_inventory_unavailable"):
        _merge()


def test_other_host_skips_inventory_but_keeps_memory_guard(host, forbid_model_load, monkeypatch):
    def no_inventory(*a, **k):
        raise AssertionError("inventory consulted on non-xwing host")

    monkeypatch.setattr(A, "build_opener", no_inventory)
    monkeypatch.setattr(A.socket, "gethostname", lambda: "other-box")
    host(2 * GIB)
    with pytest.raises(M.MergePreflightError, match="MemAvailable"):
        _merge()


def test_idle_inventory_lets_merge_proceed(host, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _FakeOpener([
        {"type": "embedding", "loaded_instances": [{"id": "embed"}]},
        {"type": "llm", "loaded_instances": []},
    ]))

    captured: dict = {}
    _stub_merge_stack(monkeypatch, captured)

    out = tmp_path / "merged"
    assert M.merge_adapter("adapter", "base", str(out), rocm=True) == str(out)
    assert out.is_dir()
    assert len(captured.get("merged_saved", [])) == 1
    assert len(captured.get("tok_saved", [])) == 1


def _stub_merge_stack(monkeypatch, captured):
    """Stub torch/peft/transformers so merge_adapter runs without a GPU or net."""
    class StubMerged:
        def save_pretrained(self, path):
            captured.setdefault("merged_saved", []).append(path)

    class StubPeft:
        def merge_and_unload(self):
            return StubMerged()

    class StubTok:
        def save_pretrained(self, path):
            captured.setdefault("tok_saved", []).append(path)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float16 = "fp16"
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: StubPeft())
    tf = types.ModuleType("transformers")

    class _FakeCausalLM:
        @staticmethod
        def from_pretrained(*a, **k):
            captured.setdefault("model_loads", []).append(dict(k))
            return object()

    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=_FakeCausalLM.from_pretrained)
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: StubTok())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", tf)


def _idle_inventory(monkeypatch):
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _FakeOpener([
        {"type": "embedding", "loaded_instances": [{"id": "embed"}]},
        {"type": "llm", "loaded_instances": []},
    ]))


def test_merge_lfm3_default_uses_auto_device_map(monkeypatch, tmp_path, host):
    """Without MERGE_DEVICE=cpu, an LFM3 merge keeps the GPU (device_map=auto)."""
    captured: dict = {}
    _stub_merge_stack(monkeypatch, captured)
    _idle_inventory(monkeypatch)
    assert M.merge_adapter("adapter", "LiquidAI/LFM3-1.7B", str(tmp_path / "merged"),
                            rocm=False) == str(tmp_path / "merged")
    assert captured["model_loads"][-1]["device_map"] == "auto"


def test_merge_lfm3_merge_device_cpu_forces_cpu_load(monkeypatch, tmp_path, host):
    """MERGE_DEVICE=cpu must force the CPU merge path even for LFM3 bases."""
    monkeypatch.setenv("MERGE_DEVICE", "cpu")
    captured: dict = {}
    _stub_merge_stack(monkeypatch, captured)
    _idle_inventory(monkeypatch)
    assert M.merge_adapter("adapter", "LiquidAI/LFM3-1.7B", str(tmp_path / "merged"),
                            rocm=False) == str(tmp_path / "merged")
    assert captured["model_loads"][-1]["device_map"] is None


def test_merge_non_lfm3_default_is_cpu(monkeypatch, tmp_path, host):
    captured: dict = {}
    _stub_merge_stack(monkeypatch, captured)
    _idle_inventory(monkeypatch)
    assert M.merge_adapter("adapter", "Qwen/Qwen2.5-7B-Instruct", str(tmp_path / "merged"),
                            rocm=False) == str(tmp_path / "merged")
    assert captured["model_loads"][-1]["device_map"] is None


def test_cli_merge_refuses_with_nonzero_code_before_load(host, forbid_model_load,
                                                          monkeypatch, tmp_path, capsys):
    from src import cli
    from src.config import Config

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("TRAIN_ADAPTER", str(adapter))
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(cli, "load", lambda *a, **k: Config({
        "train": {"model_name": "base"},
        "paths": {"lock_dir": str(tmp_path / "locks")},
    }))
    host(2 * GIB)

    assert cli.main(["cli", "merge", "--label=x"]) == 78
    assert "MemAvailable" in capsys.readouterr().out
