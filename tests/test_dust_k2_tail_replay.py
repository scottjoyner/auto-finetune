"""Synthetic equivalence tests for cached final-o_proj tail replay."""
from types import SimpleNamespace
import pytest

torch=pytest.importorskip("torch")

from experiments.dust.k2_forward_only import FinalProjectionLoRA, position_losses
from experiments.dust.k2_structured_train_compare import structured_step_batched, cosine_error
from experiments.dust.k2_tail_replay import (
    cache_base_sample, init_lora, tail_logits, tail_structured_step,
    tail_scored_structured_step,
)


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden, attn_width=12):
        super().__init__()
        self.self_attn=torch.nn.Module()
        self.self_attn.pre=torch.nn.Linear(hidden,attn_width,bias=False)
        self.self_attn.o_proj=torch.nn.Linear(attn_width,hidden,bias=False)
        self.post_attention_layernorm=torch.nn.LayerNorm(hidden)
        self.mlp=torch.nn.Sequential(
            torch.nn.Linear(hidden,hidden*2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden*2,hidden),
        )

    def forward(self,x):
        residual=x
        attn=self.self_attn.o_proj(self.self_attn.pre(x))
        hidden=residual+attn
        residual=hidden
        hidden=self.post_attention_layernorm(hidden)
        hidden=self.mlp(hidden)
        return residual+hidden


class TinyTailModel(torch.nn.Module):
    def __init__(self,vocab=32,hidden=8):
        super().__init__()
        self.embedding=torch.nn.Embedding(vocab,hidden)
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList([TinyLayer(hidden),TinyLayer(hidden)])
        self.model.norm=torch.nn.LayerNorm(hidden)
        self.lm_head=torch.nn.Linear(hidden,vocab,bias=False)

    def forward(self,input_ids,use_cache=False,return_dict=True):
        x=self.embedding(input_ids)
        for layer in self.model.layers:
            x=layer(x)
        logits=self.lm_head(self.model.norm(x))
        return SimpleNamespace(logits=logits)


def sample():
    return {
        "tokens":[1,4,8,3,12,6,9,2],
        "labels":[-100,-100,8,3,12,6,9,2],
        "assistant_tokens":6,
    }


def test_tail_replay_matches_full_forward_for_base_lora_and_jitter():
    torch.manual_seed(71)
    model=TinyTailModel().eval().requires_grad_(False)
    row=sample()
    cache=cache_base_sample(model,row,"cpu")
    a,b=init_lora(cache,rank=4,seed=59)

    base=tail_logits(model,cache,a,b)
    assert torch.allclose(base,cache.base_logits,atol=1e-6,rtol=1e-6)

    gen=torch.Generator(device="cpu").manual_seed(77)
    with torch.no_grad():
        b.copy_(torch.randn(b.shape,generator=gen)*.01)
    adapter=FinalProjectionLoRA(model,rank=4,seed=59)
    with torch.no_grad():
        adapter.a.copy_(a); adapter.b.copy_(b)
    ids=torch.tensor([row["tokens"]])
    with torch.no_grad():
        full=model(ids,use_cache=False).logits
    adapter.close()
    tail=tail_logits(model,cache,a,b)
    assert torch.allclose(tail,full,atol=1e-6,rtol=1e-5)

    jitter=torch.randn(cache.oproj_output.shape,generator=gen)*.05
    adapter=FinalProjectionLoRA(model,rank=4,seed=59)
    with torch.no_grad():
        adapter.a.copy_(a); adapter.b.copy_(b); adapter.jitter=jitter
        fullj=model(ids,use_cache=False).logits
    adapter.close()
    tailj=tail_logits(model,cache,a,b,jitter=jitter)
    assert torch.allclose(tailj,fullj,atol=1e-6,rtol=1e-5)


def test_tail_structured_update_tracks_full_batched_estimator():
    torch.manual_seed(72)
    model=TinyTailModel().eval().requires_grad_(False)
    row=sample()
    cache=cache_base_sample(model,row,"cpu")

    adapter=FinalProjectionLoRA(model,rank=4,seed=91)
    a0=adapter.a.detach().clone(); b0=adapter.b.detach().clone()
    structured_step_batched(
        model,adapter,row,"cpu",seed=1234,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    structured_step_batched(
        model,adapter,row,"cpu",seed=1235,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    full_a=adapter.a.detach().clone()-a0
    full_b=adapter.b.detach().clone()-b0
    adapter.close()

    a,b=init_lora(cache,rank=4,seed=91)
    tail_structured_step(
        model,cache,a,b,seed=1234,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    tail_structured_step(
        model,cache,a,b,seed=1235,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    tail_a=a-a0; tail_b=b-b0

    assert cosine_error(tail_a,full_a)["cosine"] > .999
    assert cosine_error(tail_b,full_b)["cosine"] > .999


def test_scored_only_tail_matches_full_batched_update():
    torch.manual_seed(73)
    model=TinyTailModel().eval().requires_grad_(False)
    row=sample()
    cache=cache_base_sample(model,row,"cpu")

    adapter=FinalProjectionLoRA(model,rank=4,seed=101)
    a0=adapter.a.detach().clone(); b0=adapter.b.detach().clone()
    first=structured_step_batched(
        model,adapter,row,"cpu",seed=2222,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    structured_step_batched(
        model,adapter,row,"cpu",seed=2223,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    full_a=adapter.a.detach().clone()-a0
    full_b=adapter.b.detach().clone()-b0
    adapter.close()

    a,b=init_lora(cache,rank=4,seed=101)
    scored_first=tail_scored_structured_step(
        model,cache,a,b,seed=2222,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    tail_scored_structured_step(
        model,cache,a,b,seed=2223,population=8,
        sigma=.05,lr=.1,direction_batch=4)
    tail_a=a-a0; tail_b=b-b0

    assert scored_first["tokens"] == row["assistant_tokens"]
    assert scored_first["train_ce"] == pytest.approx(first["train_ce"],abs=1e-6)
    assert cosine_error(tail_a,full_a)["cosine"] > .999
    assert cosine_error(tail_b,full_b)["cosine"] > .999
