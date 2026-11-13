"""Synthetic VRAM telemetry and fail-closed scaling checks; no GPU access."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src import auto_scaler as scaler


def proc(cmd, rc=0, stdout=""):
    return subprocess.CompletedProcess(cmd, rc, stdout, "")


def no_gpu_commands(monkeypatch):
    monkeypatch.setattr(
        scaler.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no GPU utility")),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    ))


def test_unknown_gpu_is_not_invented_as_15gb_8gb(monkeypatch):
    no_gpu_commands(monkeypatch)
    assert scaler.detect_vram() == (0.0, 0.0)
    with pytest.raises(ValueError, match="measured_vram_unavailable"):
        scaler.recommend(model_params_b=1.7)


def test_rocm_valid_reading_accepts_numeric_strings(monkeypatch):
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        assert "rocm-smi" == cmd[0]
        return proc(cmd, stdout=json.dumps({
            "GPU[0]": {"VRAM Total (B)": str(16 * 1024 ** 3),
                       "VRAM Used (B)": str(3 * 1024 ** 3)}
        }))
    monkeypatch.setattr(scaler.subprocess, "run", fake)
    assert scaler.detect_vram() == (16.0, 13.0)
    assert len(calls) == 1


@pytest.mark.parametrize(("used", "total"), [
    (-1, 16), (17, 16), (0, 0),
])
def test_invalid_rocm_reading_does_not_claim_free_capacity(
        monkeypatch, used, total):
    no_gpu_commands(monkeypatch)
    def fake(cmd, **kwargs):
        if cmd[0] == "rocm-smi":
            return proc(cmd, stdout=json.dumps({
                "GPU[0]": {"VRAM Total (B)": total * 1024 ** 3,
                           "VRAM Used (B)": used * 1024 ** 3}
            }))
        raise FileNotFoundError("nvidia-smi unavailable")
    monkeypatch.setattr(scaler.subprocess, "run", fake)
    assert scaler.detect_vram() == (0.0, 0.0)


def test_nvidia_fallback_uses_measured_free_not_estimate(monkeypatch):
    def fake(cmd, **kwargs):
        if cmd[0] == "rocm-smi":
            raise FileNotFoundError("no ROCm")
        if cmd[0] == "nvidia-smi":
            return proc(cmd, stdout="16384, 2048\n")
        raise AssertionError(cmd)
    monkeypatch.setattr(scaler.subprocess, "run", fake)
    assert scaler.detect_vram() == (16.0, 2.0)


@pytest.mark.parametrize("response", ["1024, 2048", "-4, 3", "NaN, 12", "bad"])
def test_untrustworthy_nvidia_memory_does_not_pass(monkeypatch, response):
    def fake(cmd, **kwargs):
        if cmd[0] == "rocm-smi":
            return proc(cmd, rc=1)
        return proc(cmd, stdout=response)
    monkeypatch.setattr(scaler.subprocess, "run", fake)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    ))
    assert scaler.detect_vram() == (0.0, 0.0)


def test_torch_fallback_reads_actual_mem_get_info(monkeypatch):
    no_gpu_commands(monkeypatch)
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda index: (4 * 1024**3, 16 * 1024**3),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert scaler.detect_vram() == (16.0, 4.0)


def test_torch_fallback_requires_actual_available_memory(monkeypatch):
    no_gpu_commands(monkeypatch)
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda index: (_ for _ in ()).throw(RuntimeError("unavailable")),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert scaler.detect_vram() == (0.0, 0.0)


@pytest.mark.parametrize(("total", "available"), [
    (0, 0), (16, 0), (0, 5), (16, -1),
    (-1, 1), (1, 2), (float("nan"), 3),
    (16, float("inf")),
])
def test_invalid_memory_cannot_generate_training_parameters(total, available):
    with pytest.raises(ValueError, match="measured_vram_unavailable"):
        scaler.recommend(total_vram_gb=total, available_vram_gb=available)


@pytest.mark.parametrize("params", [0, -1, float("nan"), float("inf")])
def test_invalid_model_size_rejected(params):
    with pytest.raises(ValueError, match="invalid_model_size"):
        scaler.recommend(total_vram_gb=16, available_vram_gb=15, model_params_b=params)


@pytest.mark.parametrize(("params", "available"), [
    (1.7, 5), (1.7, 10), (0.5, 3), (3.0, 14),
])
def test_model_state_plus_headroom_must_fit(params, available):
    with pytest.raises(ValueError, match="insufficient_vram_for_model"):
        scaler.recommend(total_vram_gb=32,
                         available_vram_gb=available, model_params_b=params)


def test_lfm3_uses_its_profile_only_after_memory_validation():
    config = scaler.recommend(available_vram_gb=20, total_vram_gb=32,
                              model_params_b=1.7, model_name="LFM3-1.7B")
    assert config.max_seq_length == 4096
    assert config.gradient_accumulation_steps == 16
    assert config.load_in_4bit is False


@pytest.mark.parametrize(("prefer_quality", "free", "expected_seq", "expected_batch"), [
    (False, 16, 128, 4),
    (False, 6.5, 128, 2),
    (False, 5.5, 64, 1),
    (False, 4.5, 32, 1),
    (True, 16, 512, 1),
    (True, 6.5, 256, 1),
    (True, 5.5, 128, 1),
    (True, 4.5, 64, 1),
])
def test_batch_vs_quality_tradeoff(prefer_quality, free, expected_seq, expected_batch):
    # 0.5B weights+optimizer+grads=3GB, plus 1GB overhead.
    config = scaler.recommend(total_vram_gb=32, available_vram_gb=free,
                              model_params_b=.5, prefer_quality=prefer_quality)
    assert config.max_seq_length == expected_seq
    assert config.per_device_batch_size == expected_batch
    assert config.throughput_estimate > 0


def test_large_model_loads_4bit_only_when_memory_really_sufficient():
    config = scaler.recommend(available_vram_gb=28, total_vram_gb=32,
                              model_params_b=3.5)
    assert config.load_in_4bit is True
    assert config.available_vram_gb == 28


def test_cli_returns_hold_without_detectable_gpu(monkeypatch, capsys):
    monkeypatch.setattr(scaler, "detect_vram", lambda: (0.0, 0.0))
    assert scaler.main(["--model-size", "1.7"]) == 78
    assert "HOLD: measured_vram_unavailable" in capsys.readouterr().out


def test_cli_json_with_measured_capacity(monkeypatch, capsys):
    monkeypatch.setattr(scaler, "detect_vram", lambda: (32.0, 25.0))
    assert scaler.main(["--model-size", "1.7", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["available_vram_gb"] == 25
    assert parsed["estimated_vram_gb"] == 32
