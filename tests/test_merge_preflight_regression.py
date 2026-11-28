"""Extra merge-preflight regressions: synthetic telemetry, no GPU/network/model load."""
from __future__ import annotations

import json
import sys
import types

import pytest

from src import merge as M
from src import training_admission as A

# TEMPORARY (PR #110 only): main received these regressions ahead of the
# merge-preflight implementation, which lands in stacked PR #111. Keeping the
# tests here but gating them on the feature being present avoids breaking #110
# while preserving every assertion; PR #111 removes this guard so the tests
# run unconditionally once the implementation is in the branch.
pytestmark = pytest.mark.skipif(
    not hasattr(A, "MEMINFO"),
    reason="merge preflight is implemented in stacked PR #111",
)

GIB = 1024 * 1024
IDLE_INVENTORY = [
    {"type": "embedding", "loaded_instances": [{"id": "embed"}]},
    {"type": "llm", "loaded_instances": []},
]
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
    """Synthetic xwing meminfo + inventory; returns setters for both."""
    meminfo = tmp_path / "meminfo"
    state = {"models": IDLE_INVENTORY}

    def set_available(kib):
        meminfo.write_text(f"MemTotal: 16 GiB\nMemAvailable: {kib} kB\n")

    def set_inventory(models):
        state["models"] = models

    set_available(8 * GIB)
    monkeypatch.setattr(A, "MEMINFO", meminfo)
    monkeypatch.setattr(A.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(A, "build_opener", lambda *a, **k: _Opener(state["models"]))
    return set_available, set_inventory


@pytest.fixture
def forbid_model_load(monkeypatch):
    """Turn any model import/load before preflight into an explicit failure."""
    # Import the real graph first: src.train imports transformers submodules.
    from src import cli, train  # noqa: F401

    def forbidden(*a, **k):
        raise AssertionError("model load started before preflight refusal")

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16 = "bf16", "fp16"
    torch.version = types.SimpleNamespace(hip=None)
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=forbidden)
    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=forbidden)
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=forbidden)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", tf)


def test_memory_floor_boundary_is_admitted(telemetry):
    """Exactly 6 GiB available passes on xwing with idle inference inventory."""
    set_available, _ = telemetry
    set_available(M.MIN_MERGE_AVAILABLE_KIB)
    assert M.preflight_merge() is None


def test_memory_one_kib_below_floor_is_refused(telemetry):
    set_available, _ = telemetry
    set_available(M.MIN_MERGE_AVAILABLE_KIB - 1)
    with pytest.raises(M.MergePreflightError, match="below the safe minimum"):
        M.preflight_merge()


@pytest.mark.parametrize("models", [
    [],
    None,
    [{"type": "llm"}],
    [{"type": "embedding", "loaded_instances": "loaded"}],
    [{"type": "unknown", "loaded_instances": []}],
])
def test_unusable_inventory_fails_closed(telemetry, models):
    """Bad telemetry must never be mistaken for an idle host."""
    _, set_inventory = telemetry
    set_inventory(models)
    with pytest.raises(M.MergePreflightError, match=r"model_inventory_(unavailable|invalid)"):
        M.preflight_merge()


def test_unreadable_meminfo_refuses_before_output_dir(telemetry, forbid_model_load, tmp_path):
    set_available, _ = telemetry
    A.MEMINFO.unlink()
    out = tmp_path / "merged"
    with pytest.raises(M.MergePreflightError, match="cannot read MemAvailable"):
        M.merge_adapter("adapter", "base", str(out))
    assert not out.exists()


def test_resident_llm_refuses_before_output_dir(telemetry, forbid_model_load, tmp_path):
    _, set_inventory = telemetry
    set_inventory(RESIDENT_LLM)
    out = tmp_path / "merged"
    with pytest.raises(M.MergePreflightError, match="resident LLM"):
        M.merge_adapter("adapter", "base", str(out))
    assert not out.exists()


def test_other_host_skips_inventory_with_headroom(telemetry, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("inventory consulted on non-xwing host")

    set_available, _ = telemetry
    monkeypatch.setattr(A.socket, "gethostname", lambda: "other-box")
    monkeypatch.setattr(A, "build_opener", forbidden)
    set_available(M.MIN_MERGE_AVAILABLE_KIB)
    assert M.preflight_merge() is None


def test_cli_merge_refuses_resident_llm_with_78(telemetry, forbid_model_load,
                                                 monkeypatch, tmp_path, capsys):
    from src import cli
    from src.config import Config

    _, set_inventory = telemetry
    set_inventory(RESIDENT_LLM)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("TRAIN_ADAPTER", str(adapter))
    monkeypatch.setenv("TRAIN_MERGED_OUT", str(tmp_path / "merged"))
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(cli, "load", lambda *a, **k: Config({
        "train": {"model_name": "base"},
        "paths": {"lock_dir": str(tmp_path / "locks")},
    }))
    assert cli.main(["cli", "merge", "--label=x"]) == 78
    assert "resident LLM" in capsys.readouterr().out
    assert not (tmp_path / "merged").exists()
