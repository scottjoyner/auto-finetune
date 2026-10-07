"""CPU-testable frozen K2-Horizon last-layer forward-only estimation guards."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
# Import dynamically loaded HF dependencies before monkeypatching autograd methods;
# torch._dynamo scans the function registry during import.
from transformers import AutoModelForCausalLM  # noqa: F401
import torch._dynamo  # noqa: F401

from experiments.dust.k2_forward_only import main, position_losses, run

MODEL_DIR = Path("/media/scott/data/finetune-staging/models/K2-Horizon-0.9B")


def test_causal_shift_and_prefix_mask_are_position_aligned():
    logits = torch.zeros((1, 5, 7), dtype=torch.float32)
    labels = torch.tensor([[1, -100, 3, 2, -100]])
    logits[0, 1, 3] = 4.0
    logits[0, 2, 2] = 4.0
    losses, valid = position_losses(logits, labels)
    assert valid.tolist() == [[False, True, True, False, False]]
    assert losses.shape == (1, 5)
    assert losses[0, 1] > 0 and losses[0, 2] > 0
    assert torch.count_nonzero(losses[~valid]).item() == 0
    expected = torch.nn.functional.cross_entropy(
        logits[:, 1:3].reshape(-1, 7), torch.tensor([3, 2]), reduction="none"
    )
    assert torch.allclose(losses[0, 1:3], expected, atol=1e-6)


def test_all_masked_targets_are_rejected():
    with pytest.raises(ValueError, match="No unmasked"):
        position_losses(torch.zeros(1, 3, 7), torch.full((1, 3), -100))


def test_hardware_and_hyperparameter_gates_before_any_model_load(tmp_path):
    for kwargs in ({"steps": 3}, {"draws": 129}, {"rank": 0},
                   {"sigma": 0}, {"lr": 0}, {"device": "npu"}):
        with pytest.raises(ValueError):
            run(model_dir=tmp_path, **kwargs)
    with pytest.raises(ValueError, match="requires CPU"):
        run(model_dir=tmp_path, tiny=False, device="cuda", draws=2)
    with pytest.raises(ValueError, match="draw limit"):
        run(model_dir=tmp_path, tiny=False, draws=5)


def test_true_k2_architecture_forward_only_tiny_cpu_and_no_backward(monkeypatch):
    if not MODEL_DIR.is_dir():
        pytest.skip("K2 configuration not installed on test host")
    def fail(*args, **kwargs):
        raise AssertionError("Autograd operation unexpectedly called")

    monkeypatch.setattr(torch.Tensor, "backward", fail)
    monkeypatch.setattr(torch.autograd, "backward", fail)
    monkeypatch.setattr(torch.autograd, "grad", fail)
    result = run(model_dir=MODEL_DIR, device="cpu", tiny=True,
                 draws=16, steps=1, seed=42)
    assert result["architecture"] == "K2HorizonForCausalLM"
    assert result["tiny_random_model"] is True
    assert result["model_base_unchanged"] is True
    assert result["all_model_grads_absent"] is True
    assert result["backward_calls"] == 0
    assert result["adapter_b_norm"] > 0
    assert all(math.isfinite(x) for x in result["loss_history"])
    assert result["checkpoint_written"] is False


def test_output_creates_only_new_metadata(monkeypatch, tmp_path):
    if not MODEL_DIR.is_dir():
        pytest.skip("K2 configuration not installed on test host")
    artifact = tmp_path / "smoke.json"
    argv = ["--model-dir", str(MODEL_DIR), "--draws", "8",
            "--steps", "1", "--output", str(artifact)]
    assert main(argv) == 0
    snapshot = artifact.read_bytes()
    record = json.loads(snapshot)
    assert record["checkpoint_written"] is False
    assert record["real_dataset_used"] is False
    with pytest.raises(SystemExit):
        main(argv)
    assert snapshot == artifact.read_bytes()
