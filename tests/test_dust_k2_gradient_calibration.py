"""Synthetic functional tests for K2 last-o_proj gradient calibration."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.dust.k2_gradient_calibration import calibrate, summarize


class LastLayer(torch.nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class PositionLocalCausal(torch.nn.Module):
    def __init__(self, vocab=24, hidden=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([LastLayer(hidden)])
        self.mlp = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, use_cache=False, return_dict=True):
        x = self.embedding(input_ids)
        x = self.model.layers[-1].self_attn.o_proj(x)
        x = torch.tanh(self.mlp(x))
        return SimpleNamespace(logits=self.head(x))


def test_antithetic_estimator_tracks_exact_local_gradient_and_freezes_base():
    torch.manual_seed(101)
    model = PositionLocalCausal().eval().requires_grad_(False)
    original = model.model.layers[-1].self_attn.o_proj.weight.clone()
    sample = {
        "tokens": [1, 4, 8, 3, 12, 6, 9, 2],
        "labels": [-100, -100, 8, 3, 12, 6, 9, 2],
        "assistant_tokens": 6,
    }
    result = calibrate(
        model, sample, device="cpu", seeds=[7], sigmas=[0.025],
        populations=[16, 64, 128], rank=4, max_seconds=30)
    anti = [p for p in result["points"]
            if p["estimator"] == "antithetic" and p["population"] == 128][0]
    assert anti["output_gradient"]["cosine"] > 0.75
    assert anti["lora_b_gradient"]["cosine"] > 0.75
    assert result["base_unchanged"] is True
    assert result["base_model_grads_absent"] is True
    assert result["checkpoint_written"] is False
    assert torch.equal(original, model.model.layers[-1].self_attn.o_proj.weight)


def test_summary_aggregates_seed_results_without_raw_data():
    rows = [
        {"estimator": "antithetic", "sigma": .05, "population": 32,
         "output_gradient": {"cosine": .8, "relative_l2_error": .5},
         "lora_b_gradient": {"cosine": .9, "relative_l2_error": .4}},
        {"estimator": "antithetic", "sigma": .05, "population": 32,
         "output_gradient": {"cosine": .6, "relative_l2_error": .7},
         "lora_b_gradient": {"cosine": .7, "relative_l2_error": .6}},
    ]
    result = summarize(rows)
    assert len(result) == 1
    assert result[0]["seeds"] == 2
    assert result[0]["mean_output_cosine"] == pytest.approx(.7)
    assert result[0]["mean_lora_b_cosine"] == pytest.approx(.8)


def test_full_orthogonal_basis_recovers_local_gradient_in_tiny_hidden_space():
    torch.manual_seed(202)
    model = PositionLocalCausal(hidden=8).eval().requires_grad_(False)
    sample = {
        "tokens": [1, 5, 7, 3, 11, 4],
        "labels": [-100, -100, 7, 3, 11, 4],
        "assistant_tokens": 4,
    }
    result = calibrate(
        model, sample, device="cpu", seeds=[9], sigmas=[0.01],
        populations=[8], rank=4, max_seconds=30, direction_mode="orthogonal")
    anti = [p for p in result["points"] if p["estimator"] == "antithetic"][0]
    assert anti["output_gradient"]["cosine"] > 0.95
    assert anti["lora_b_gradient"]["cosine"] > 0.95
    assert result["direction_mode"] == "orthogonal"
