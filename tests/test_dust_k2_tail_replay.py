"""Synthetic equivalence tests for cached final-o_proj tail replay."""
from types import SimpleNamespace
import pytest

torch=pytest.importorskip("torch")

import experiments.dust.k2_structured_train_compare as structured_module
import experiments.dust.k2_tail_replay as tail_module
from experiments.dust.k2_forward_only import FinalProjectionLoRA, position_losses
from experiments.dust.k2_structured_train_compare import structured_step_batched, cosine_error
from experiments.dust.k2_tail_replay import (
    cache_base_sample, init_lora, tail_logits, tail_structured_step,
    tail_scored_structured_step, tail_scored_structured_estimate,
    tail_scored_tokenwise_estimate, tail_scored_tokenwise_step,
    local_lora_grads_scored, score_tail_scored,
)


@pytest.fixture(autouse=True)
def synthetic_memory_headroom(monkeypatch):
    """Synthetic unit tests must not depend on current fleet memory pressure."""
    enough=8*1024**3
    monkeypatch.setattr(structured_module,"read_mem_available",lambda: enough)
    monkeypatch.setattr(tail_module,"memory_available_bytes",lambda: enough)


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


def test_tokenwise_antithetic_recovers_exact_local_output_and_lora_gradients():
    torch.manual_seed(74)
    model=TinyTailModel().eval().requires_grad_(False)
    row=sample()
    cache=cache_base_sample(model,row,"cpu")
    a,b=init_lora(cache,rank=4,seed=111)
    gen=torch.Generator(device="cpu").manual_seed(112)
    with torch.no_grad():
        b.copy_(torch.randn(b.shape,generator=gen)*.02)

    positions,_=score_tail_scored(model,cache,a,b)
    # Exact local output gradient at supervised final-o_proj positions.
    from experiments.dust.k2_tail_replay import supervised_source_positions
    scored_positions,_=supervised_source_positions(cache)
    jitter=torch.zeros(
        1,scored_positions.numel(),cache.oproj_output.shape[-1],
        dtype=torch.float32,requires_grad=True)
    losses,_=score_tail_scored(model,cache,a,b,jitter=jitter)
    exact=torch.autograd.grad(losses.sum(),jitter)[0].detach()

    estimate=tail_scored_tokenwise_estimate(
        model,cache,a,b,seed=555,population=4096,
        sigma=.005,direction_batch=32)
    out=cosine_error(estimate["estimate"],exact)
    exact_a,exact_b=local_lora_grads_scored(
        cache,a,b,scored_positions,exact,count=int(scored_positions.numel()))
    est_a,est_b=local_lora_grads_scored(
        cache,a,b,scored_positions,estimate["estimate"],
        count=int(scored_positions.numel()))
    assert out["cosine"] > .97
    assert cosine_error(est_a,exact_a)["cosine"] > .97
    assert cosine_error(est_b,exact_b)["cosine"] > .97
    assert estimate["unique_direction_vectors"] == 4096 * row["assistant_tokens"]
    assert estimate["token_position_perturbations"] == 4096 * row["assistant_tokens"]
    assert estimate["independent_noise_per_token"] is True


def test_tokenwise_step_is_forward_only_and_updates_fresh_lora():
    torch.manual_seed(75)
    model=TinyTailModel().eval().requires_grad_(False)
    row=sample()
    cache=cache_base_sample(model,row,"cpu")
    a,b=init_lora(cache,rank=4,seed=121)
    a0=a.clone(); b0=b.clone()
    first=tail_scored_tokenwise_step(
        model,cache,a,b,seed=777,population=256,
        sigma=.02,lr=.1,direction_batch=16)
    assert first["tokens"] == row["assistant_tokens"]
    assert first["tail_forward_calls"] == 17
    assert first["unique_direction_vectors"] == 256 * row["assistant_tokens"]
    assert first["token_position_perturbations"] == 256 * row["assistant_tokens"]
    assert first["independent_noise_per_token"] is True
    assert torch.equal(a,a0)  # B starts at zero, so first A gradient is zero.
    assert not torch.equal(b,b0)
    assert all(p.grad is None for p in model.parameters())
