"""CPU-only tests for the bounded synthetic Dust-inspired LoRA estimator."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from experiments.dust.zo_lora_probe import main, run


def test_fixed_seed_decreases_synthetic_loss_and_matches_exact_gradient():
    a = run(device="cpu", seed=42, draws=128, steps=6)
    b = run(device="cpu", seed=42, draws=128, steps=6)
    assert a["loss_history"] == b["loss_history"]
    assert a["initial_loss"] > a["final_loss"] > 0
    assert min(a["gradient_cosines"]) > 0.95
    assert a["backward_calls"] == 0
    assert a["checkpoint_written"] is False
    assert a["not_k2_training"] is True


def test_no_backward_even_if_torch_autograd_methods_raise(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Autograd backward was called")
    monkeypatch.setattr(torch.Tensor, "backward", fail)
    monkeypatch.setattr(torch.autograd, "backward", fail)
    monkeypatch.setattr(torch.autograd, "grad", fail)
    result = run(device="cpu", seed=7, draws=64, steps=3)
    assert result["final_loss"] < result["initial_loss"]


def test_bounded_parameters_do_not_oversubscribe():
    for kwargs in ({"draws": 513}, {"steps": 21}, {"draws": 0},
                   {"sigma": 0}, {"lr": 0}, {"device": "mps"}):
        with pytest.raises(ValueError):
            run(**kwargs)

def test_optional_output_must_be_new_and_is_synthetic_only(tmp_path):
    evidence = tmp_path / "probe.json"
    args = ["--device", "cpu", "--seed", "3", "--draws", "128",
            "--steps", "3", "--output", str(evidence)]
    assert main(args) == 0
    artifact = json.loads(evidence.read_text())
    assert artifact["checkpoint_written"] is False
    assert artifact["kind"] == "synthetic_only_forward_zeroth_order_lora_probe"
    unchanged = evidence.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert unchanged == evidence.read_bytes()
