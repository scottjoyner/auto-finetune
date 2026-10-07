"""Synthetic-only functional tests for the matched estimator/control harness."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from experiments.dust.k2_matched_compare import (
    one_backend, loss_on_heldout, read_mem_available, run_pilot,
)


class MinimalLastLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(12, 12, bias=False)


class MinimalCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 12)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([MinimalLastLayer()])
        self.head = torch.nn.Linear(12, 32)

    def forward(self, input_ids, use_cache=False, return_dict=True):
        x = self.embedding(input_ids)
        x = torch.tanh(self.model.layers[-1].self_attn.o_proj(x))
        return SimpleNamespace(logits=self.head(x))


def pair(ids):
    return {"tokens": ids, "labels": [-100, -100] + ids[2:]}


def test_matched_control_with_independent_pretrained_weight_guards():
    torch.manual_seed(11)
    model = MinimalCausal().eval().requires_grad_(False)
    train = [pair([1, 4, 9, 3, 12, 2, 8, 0])]
    eval_data = [pair([1, 7, 2, 14, 5, 9, 8, 0])]
    before = loss_on_heldout(model, eval_data, "cpu")
    old_weight = model.model.layers[-1].self_attn.o_proj.weight.clone()
    grad = one_backend(model, name="backprop", train=train, eval_rows=eval_data,
                       device="cpu", seed=11, steps=2, draws=16,
                       sigma=0.05, lr=0.25, max_seconds=30)
    dust = one_backend(model, name="dust", train=train, eval_rows=eval_data,
                       device="cpu", seed=11, steps=2, draws=16,
                       sigma=0.05, lr=0.25, max_seconds=30)
    assert grad["initial_holdout"] == dust["initial_holdout"] == before
    assert grad["backward_calls"] == 2 and dust["backward_calls"] == 0
    assert dust["checkpoint_written"] is False
    assert grad["base_unchanged"] and dust["base_unchanged"]
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(old_weight, model.model.layers[-1].self_attn.o_proj.weight)


def test_read_only_failure_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="source must differ"):
        run_pilot(model_dir=tmp_path, train_file=tmp_path/"same.jsonl",
                  eval_file=tmp_path/"same.jsonl", device="cpu",
                  expected_sha="unimportant")
    assert read_mem_available() > 0
