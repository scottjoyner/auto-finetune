"""Synthetic tests for orthogonal-antithetic K2 adapter training fidelity."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.dust.k2_structured_train_compare import cosine_error, run_backend


class LastLayer(torch.nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=24, hidden=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([LastLayer(hidden)])
        self.mlp = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, use_cache=False, return_dict=True):
        x=self.embedding(input_ids)
        x=self.model.layers[-1].self_attn.o_proj(x)
        x=torch.tanh(self.mlp(x))
        return SimpleNamespace(logits=self.head(x))


def sample(values):
    return {"tokens":values, "labels":[-100,-100]+values[2:],
            "assistant_tokens":len(values)-2}


def test_full_basis_structured_updates_track_two_step_backprop():
    torch.manual_seed(17)
    model=TinyModel().eval().requires_grad_(False)
    frozen=model.model.layers[-1].self_attn.o_proj.weight.clone()
    train=[
        sample([1,4,8,3,12,6,9,2]),
        sample([1,7,5,11,3,8,6,2]),
    ]
    held=[sample([1,6,10,2,13,4,7,2])]
    exact=run_backend(
        model,backend="backprop",train=train,heldout=held,device="cpu",
        seed=9,steps=2,population=8,sigma=.01,lr=.1,max_seconds=30)
    structured=run_backend(
        model,backend="orthogonal_antithetic",train=train,heldout=held,
        device="cpu",seed=9,steps=2,population=8,sigma=.01,lr=.1,
        max_seconds=30)
    a=cosine_error(
        structured["final_a"]-structured["initial_a"],
        exact["final_a"]-exact["initial_a"])
    b=cosine_error(
        structured["final_b"]-structured["initial_b"],
        exact["final_b"]-exact["initial_b"])
    assert a["cosine"] > .9
    assert b["cosine"] > .95
    assert exact["metrics"]["backward_calls"] == 2
    assert structured["metrics"]["backward_calls"] == 0
    assert torch.equal(frozen,model.model.layers[-1].self_attn.o_proj.weight)
