"""Offline, deterministic coverage for ``src.auto_scaler``.

Only the pure / hermetic surface is exercised here:

* the ``ScalingConfig`` dataclass and its ``asdict`` JSON-shape formatting,
* the decision math in ``recommend()`` (VRAM -> sequence length, batch size,
  gradient accumulation, 4-bit flag, throughput estimate; LFM3 special case;
  negative-free-VRAM fallback; exact ``>`` boundaries), and
* the output formatting in ``main()`` with ``detect_vram`` replaced by a fake.

No network, GPU, DB, subprocess, or wall-clock sleeps are used. ``recommend``
is always called with explicit VRAM values, and ``main`` is driven with a
monkeypatched ``detect_vram`` so no ``rocm-smi`` / ``nvidia-smi`` / torch call
can run. GPU-detection internals are deliberately not covered.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

import pytest

from src import auto_scaler as AS
from src.auto_scaler import ScalingConfig, recommend


def rec(**kwargs):
    """Call recommend with explicit VRAM so detect_vram is never reached."""
    kwargs.setdefault("available_vram_gb", 16.0)
    kwargs.setdefault("total_vram_gb", 16.0)
    return recommend(**kwargs)


# --- ScalingConfig: pure data container + JSON formatting shape ----------------

def test_scaling_config_defaults():
    c = ScalingConfig()
    assert is_dataclass(c)
    assert c.max_seq_length == 64
    assert c.per_device_batch_size == 1
    assert c.gradient_accumulation_steps == 8
    assert c.load_in_4bit is False
    assert c.estimated_vram_gb == 0.0
    assert c.available_vram_gb == 0.0
    assert c.throughput_estimate == 0.0


def test_scaling_config_asdict_is_json_serializable():
    c = ScalingConfig(
        max_seq_length=128, per_device_batch_size=2,
        gradient_accumulation_steps=4, load_in_4bit=True,
        estimated_vram_gb=24.0, available_vram_gb=16.0,
        throughput_estimate=4.0,
    )
    d = asdict(c)
    assert d == {
        "max_seq_length": 128,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "load_in_4bit": True,
        "estimated_vram_gb": 24.0,
        "available_vram_gb": 16.0,
        "throughput_estimate": 4.0,
    }
    assert json.loads(json.dumps(d)) == d


# --- recommend(): hermetic guard and decision matrix ---------------------------

def test_recommend_does_not_detect_when_vram_supplied(monkeypatch):
    def boom():
        raise AssertionError("detect_vram must not be called with explicit VRAM")
    monkeypatch.setattr(AS, "detect_vram", boom)
    c = recommend(available_vram_gb=16.0, total_vram_gb=16.0)
    assert c.max_seq_length == 128
    assert c.per_device_batch_size == 4


@pytest.mark.parametrize(
    "avail, params, prefer_quality, seq, batch, grad, fourbit, tput",
    [
        # batch-first branches (>4, >2, >1, else)
        (16.0, 0.5, False, 128, 4, 2, False, 4.0),   # free=12 -> >4
        (8.0, 0.5, False, 128, 2, 4, False, 4.0),    # free==4 boundary -> >2
        (7.0, 0.5, False, 128, 2, 4, True, 4.0),     # free=3, low-VRAM 4-bit
        (6.0, 0.5, False, 64, 1, 8, True, 8.0),      # free==2 boundary -> >1
        (5.5, 0.5, False, 64, 1, 8, True, 8.0),      # free=1.5
        (5.0, 0.5, False, 32, 1, 16, True, 16.0),    # free==1 boundary -> else
        (4.0, 0.5, False, 32, 1, 16, True, 16.0),    # free=0
        (3.0, 0.5, False, 32, 1, 16, True, 16.0),    # free<0 -> fallback 0.9
        # quality-first branches (>4, >2, >1, else)
        (16.0, 0.5, True, 512, 1, 8, False, 1.0),    # free=12 -> >4
        (8.0, 0.5, True, 256, 1, 8, False, 2.0),     # free==4 boundary -> >2
        (7.0, 0.5, True, 256, 1, 8, True, 2.0),      # free=3
        (6.0, 0.5, True, 128, 1, 8, True, 4.0),      # free==2 boundary -> >1
        (5.5, 0.5, True, 128, 1, 8, True, 4.0),      # free=1.5
        (5.0, 0.5, True, 64, 1, 8, True, 8.0),       # free==1 boundary -> else
        # large model triggers 4-bit even with ample VRAM
        (30.0, 4.0, False, 128, 4, 2, True, 4.0),
    ],
)
def test_recommend_decision_matrix(avail, params, prefer_quality,
                                   seq, batch, grad, fourbit, tput):
    c = rec(available_vram_gb=avail, model_params_b=params,
            prefer_quality=prefer_quality)
    assert c.max_seq_length == seq
    assert c.per_device_batch_size == batch
    assert c.gradient_accumulation_steps == grad
    assert c.load_in_4bit is fourbit
    assert c.throughput_estimate == pytest.approx(tput)
    assert c.estimated_vram_gb == 16.0
    assert c.available_vram_gb == avail


def test_negative_free_vram_fallback_still_clamps_to_smallest_config():
    # model_mem=18, avail=10 -> free=-9 < 0 -> fallback 10*0.3=3.0 (>2)
    c = rec(available_vram_gb=10.0, model_params_b=3.0)
    assert c.max_seq_length == 128
    assert c.per_device_batch_size == 2
    assert c.gradient_accumulation_steps == 4
    assert c.throughput_estimate == pytest.approx(4.0)


# --- LFM3 special case ---------------------------------------------------------

def test_lfm3_special_case_overrides_low_vram_fourbit_rule():
    c = rec(available_vram_gb=6.0, total_vram_gb=16.0,
            model_params_b=1.7, model_name="LFM3-1.7B")
    assert c.max_seq_length == 4096
    assert c.per_device_batch_size == 1
    assert c.gradient_accumulation_steps == 16
    assert c.load_in_4bit is False  # early return ignores the <8GB 4-bit rule
    assert c.throughput_estimate == pytest.approx(2.0)
    assert c.estimated_vram_gb == 16.0
    assert c.available_vram_gb == 6.0


@pytest.mark.parametrize("name", ["lfm3-1.7b", "LfM3-Instruct", "some-LFM3-model"])
def test_lfm3_name_match_is_case_insensitive(name):
    c = rec(available_vram_gb=16.0, model_params_b=1.7, model_name=name)
    assert c.max_seq_length == 4096
    assert c.load_in_4bit is False
    assert c.throughput_estimate == pytest.approx(2.0)


def test_lfm3_special_case_requires_params_below_two():
    c = rec(available_vram_gb=30.0, total_vram_gb=30.0,
            model_params_b=2.0, model_name="lfm3-2b")
    assert c.max_seq_length == 128
    assert c.per_device_batch_size == 4
    assert c.load_in_4bit is False


# --- main(): deterministic output formatting (detect_vram faked) --------------

def test_main_json_output_is_deterministic(monkeypatch, capsys):
    monkeypatch.setattr(AS, "detect_vram", lambda: (16.0, 16.0))
    assert AS.main(["--model-size", "0.5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_seq_length"] == 128
    assert payload["per_device_batch_size"] == 4
    assert payload["gradient_accumulation_steps"] == 2
    assert payload["load_in_4bit"] is False
    assert payload["estimated_vram_gb"] == 16.0
    assert payload["available_vram_gb"] == 16.0
    assert payload["throughput_estimate"] == pytest.approx(4.0)


def test_main_human_output_contains_env_var_formatting(monkeypatch, capsys):
    monkeypatch.setattr(AS, "detect_vram", lambda: (8.0, 8.0))
    assert AS.main(["--model-size", "0.5", "--prefer-quality"]) == 0
    out = capsys.readouterr().out
    assert "[autoscale] VRAM: 8.0GB available / 8.0GB total" in out
    assert "max_seq_length: 256" in out
    assert "batch_size: 1" in out
    assert "grad_accum: 8" in out
    assert "4-bit: False" in out
    assert "throughput estimate: 2.00" in out
    assert "TRAIN_MAX_SEQ_LENGTH=256" in out
    assert "TRAIN_PER_DEVICE_BATCH_SIZE=1" in out
    assert "TRAIN_GRADIENT_ACCUMULATION_STEPS=8" in out
    assert "TRAIN_LOAD_4BIT=0" in out
