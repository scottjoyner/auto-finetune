"""Cache the frozen K2 prefix and replay only the final o_proj downstream tail.

Valid only for adapters/perturbations attached to the final decoder layer's
attention o_proj output. Earlier layers and q/k/v are explicitly out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time


def memory_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


@dataclass
class TailCache:
    layer_input: object
    oproj_input: object
    oproj_output: object
    labels: object
    base_logits: object


class Capture:
    def __init__(self, model):
        self.layer = model.model.layers[-1]
        self.oproj = self.layer.self_attn.o_proj
        self.layer_input = None
        self.oproj_input = None
        self.oproj_output = None
        self.pre = self.layer.register_forward_pre_hook(self._pre)
        self.out = self.oproj.register_forward_hook(self._out)

    def _pre(self, _module, args):
        self.layer_input = args[0].detach()

    def _out(self, _module, args, output):
        self.oproj_input = args[0].detach()
        self.oproj_output = output.detach()

    def close(self):
        self.pre.remove()
        self.out.remove()


def cache_base_sample(
    model, sample: dict, device: str, *, keep_logits: bool = True
) -> TailCache:
    import torch

    ids=torch.tensor([sample["tokens"]],device=device,dtype=torch.long)
    labels=torch.tensor([sample["labels"]],device=device,dtype=torch.long)
    cap=Capture(model)
    try:
        with torch.no_grad():
            logits=model(input_ids=ids,use_cache=False,return_dict=True).logits
        if cap.layer_input is None or cap.oproj_input is None or cap.oproj_output is None:
            raise AssertionError("Failed to capture final-layer tail state")
        return TailCache(
            layer_input=cap.layer_input.clone(),
            oproj_input=cap.oproj_input.clone(),
            oproj_output=cap.oproj_output.clone(),
            labels=labels,
            base_logits=(logits.detach().clone() if keep_logits else None),
        )
    finally:
        cap.close()


def init_lora(cache: TailCache, *, rank: int, seed: int):
    import torch
    if rank not in (2,4,8):
        raise ValueError("rank must be 2/4/8")
    in_features=cache.oproj_input.shape[-1]
    out_features=cache.oproj_output.shape[-1]
    gen=torch.Generator(device="cpu").manual_seed(seed)
    a=(torch.randn(rank,in_features,generator=gen)/math.sqrt(in_features)).to(
        cache.oproj_input.device)
    b=torch.zeros(out_features,rank,device=cache.oproj_input.device)
    return a,b


def tail_logits(model, cache: TailCache, a, b, jitter=None):
    """Replay exactly from final o_proj output through final MLP/norm/lm_head."""
    x=cache.oproj_input
    update=(x.float() @ a.T) @ b.T
    if jitter is not None:
        update=update + jitter
    attn=cache.oproj_output + update.to(dtype=cache.oproj_output.dtype)

    batch=attn.shape[0]
    residual0=cache.layer_input
    if residual0.shape[0] != batch:
        residual0=residual0.expand(batch,-1,-1)
    hidden=residual0 + attn
    layer=model.model.layers[-1]
    residual1=hidden
    hidden=layer.post_attention_layernorm(hidden)
    hidden=layer.mlp(hidden)
    if isinstance(hidden,tuple):
        hidden=hidden[0]
    hidden=residual1 + hidden
    hidden=model.model.norm(hidden)
    return model.lm_head(hidden)


def score_tail(model, cache: TailCache, a, b, jitter=None):
    from experiments.dust.k2_forward_only import position_losses
    logits=tail_logits(model,cache,a,b,jitter=jitter)
    labels=cache.labels
    if labels.shape[0] != logits.shape[0]:
        labels=labels.expand(logits.shape[0],-1)
    return position_losses(logits,labels)


def apply_local_lora_update(cache: TailCache, a, b, output_gradient, *, count: int, lr: float):
    import torch
    with torch.no_grad():
        x=cache.oproj_input.reshape(-1,cache.oproj_input.shape[-1]).float()
        dy=output_gradient.reshape(-1,output_gradient.shape[-1]).float()
        z=x @ a.T
        grad_b=dy.T @ z / count
        grad_a=(dy @ b).T @ x / count
        b.add_(grad_b,alpha=-lr)
        a.add_(grad_a,alpha=-lr)
    return grad_a,grad_b


def tail_structured_step(model, cache: TailCache, a, b, *, seed: int,
                         population: int, sigma: float, lr: float,
                         direction_batch: int) -> dict:
    import torch
    if population > cache.oproj_output.shape[-1]:
        raise ValueError("population exceeds hidden width")
    if direction_batch not in (1,2,4,8,16,32):
        raise ValueError("unsupported direction_batch")
    if not (0 < sigma <= .5 and 0 < lr <= 1.0):
        raise ValueError("invalid sigma/lr")

    with torch.no_grad():
        clean,mask=score_tail(model,cache,a,b)
        count=int(mask.sum().item())
        if count <= 0 or not bool(torch.isfinite(clean).all()):
            raise ArithmeticError("invalid clean tail loss")
        hidden=cache.oproj_output.shape[-1]
        seq=cache.oproj_output.shape[1]
        gen=torch.Generator(device=cache.oproj_output.device).manual_seed(seed)
        raw=torch.randn(hidden,population,generator=gen,
                        device=cache.oproj_output.device,dtype=torch.float32)
        directions=(torch.linalg.qr(raw,mode="reduced").Q.T*math.sqrt(hidden)).contiguous()
        del raw
        estimate=torch.zeros(
            1,seq,hidden,device=cache.oproj_output.device,dtype=torch.float32)
        forwards=1
        min_host_mem=memory_available_bytes()
        if min_host_mem and min_host_mem < 1536 * 1024**2:
            raise MemoryError("host memory below tail-replay safety floor")
        started=time.monotonic()
        for start in range(0,population,direction_batch):
            available=memory_available_bytes()
            if available:
                min_host_mem=min(min_host_mem,available)
                if available < 1536 * 1024**2:
                    raise MemoryError("host memory below tail-replay safety floor")
            d=directions[start:start+direction_batch]
            width=d.shape[0]
            signed=torch.cat((d,-d),dim=0)
            jitter=signed[:,None,:].expand(2*width,seq,hidden)*sigma
            losses,masks=score_tail(model,cache,a,b,jitter=jitter)
            plus,minus=losses[:width].float(),losses[width:].float()
            expected=mask.expand(width,-1)
            if not torch.equal(masks[:width],expected) or not torch.equal(masks[width:],expected):
                raise AssertionError("tail perturbation changed causal mask")
            diff=(plus-minus)*expected.float()
            estimate += (
                diff.unsqueeze(-1)*d[:,None,:]
            ).sum(dim=0,keepdim=True)/(2.0*sigma*population)
            forwards += 1
        grad_a,grad_b=apply_local_lora_update(
            cache,a,b,estimate,count=count,lr=lr)
        return {
            "train_ce":float(clean.sum().item()/count),
            "tokens":count,
            "population":population,
            "direction_batch":direction_batch,
            "tail_forward_calls":forwards,
            "elapsed_seconds":time.monotonic()-started,
            "grad_a_norm":float(grad_a.norm().item()),
            "grad_b_norm":float(grad_b.norm().item()),
            "min_host_mem_available_bytes":min_host_mem,
        }


def supervised_source_positions(cache: TailCache):
    """Return source-logit positions and next-token targets for supervised CE."""
    import torch
    shifted=cache.labels[:,1:]
    valid=shifted.ne(-100)
    if valid.shape[0] != 1:
        raise ValueError("Tail cache must represent one source sample")
    positions=torch.nonzero(valid[0],as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("No supervised next-token positions")
    targets=shifted[0,positions]
    return positions,targets


def tail_scored_hidden(model, cache: TailCache, a, b, positions, jitter=None):
    """Replay only supervised token positions after final o_proj."""
    x=cache.oproj_input[:,positions,:]
    update=(x.float() @ a.T) @ b.T
    if jitter is not None:
        update=update + jitter
    attn=cache.oproj_output[:,positions,:] + update.to(
        dtype=cache.oproj_output.dtype)
    batch=attn.shape[0]
    residual0=cache.layer_input[:,positions,:]
    if residual0.shape[0] != batch:
        residual0=residual0.expand(batch,-1,-1)
    hidden=residual0 + attn
    layer=model.model.layers[-1]
    residual1=hidden
    hidden=layer.post_attention_layernorm(hidden)
    hidden=layer.mlp(hidden)
    if isinstance(hidden,tuple):
        hidden=hidden[0]
    hidden=residual1 + hidden
    return model.model.norm(hidden)


def score_tail_scored(model, cache: TailCache, a, b, jitter=None):
    """Token CE for supervised positions only; no masked-position LM head work."""
    import torch
    from torch.nn import functional as F
    positions,targets=supervised_source_positions(cache)
    hidden=tail_scored_hidden(
        model,cache,a,b,positions,jitter=jitter)
    logits=model.lm_head(hidden).float()
    batch=logits.shape[0]
    target=targets.unsqueeze(0).expand(batch,-1)
    losses=F.cross_entropy(
        logits.reshape(-1,logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape(batch,-1)
    if not bool(torch.isfinite(losses).all()):
        raise ArithmeticError("nonfinite scored tail CE")
    return losses,positions


def apply_local_lora_update_scored(
    cache: TailCache, a, b, positions, output_gradient, *, count: int, lr: float
):
    import torch
    with torch.no_grad():
        x=cache.oproj_input[:,positions,:].reshape(
            -1,cache.oproj_input.shape[-1]).float()
        dy=output_gradient.reshape(-1,output_gradient.shape[-1]).float()
        z=x @ a.T
        grad_b=dy.T @ z / count
        grad_a=(dy @ b).T @ x / count
        b.add_(grad_b,alpha=-lr)
        a.add_(grad_a,alpha=-lr)
    return grad_a,grad_b


def tail_scored_structured_step(
    model, cache: TailCache, a, b, *, seed: int, population: int,
    sigma: float, lr: float, direction_batch: int,
) -> dict:
    """Orthogonal antithetic update using only supervised final-tail positions."""
    import torch
    positions,targets=supervised_source_positions(cache)
    count=int(positions.numel())
    out_features=cache.oproj_output.shape[-1]
    if population > out_features:
        raise ValueError("population exceeds final o_proj output width")
    if direction_batch not in (1,2,4,8,16,32):
        raise ValueError("unsupported direction_batch")
    if not (0 < sigma <= .5 and 0 < lr <= 1.0):
        raise ValueError("invalid sigma/lr")

    with torch.no_grad():
        clean,_=score_tail_scored(model,cache,a,b)
        gen=torch.Generator(device=cache.oproj_output.device).manual_seed(seed)
        raw=torch.randn(
            out_features,population,generator=gen,
            device=cache.oproj_output.device,dtype=torch.float32)
        directions=(
            torch.linalg.qr(raw,mode="reduced").Q.T*math.sqrt(out_features)
        ).contiguous()
        del raw
        estimate=torch.zeros(
            1,count,out_features,device=cache.oproj_output.device,
            dtype=torch.float32)
        forwards=1
        min_host_mem=memory_available_bytes()
        if min_host_mem and min_host_mem < 1536 * 1024**2:
            raise MemoryError("host memory below scored-tail safety floor")
        started=time.monotonic()
        for start in range(0,population,direction_batch):
            available=memory_available_bytes()
            if available:
                min_host_mem=min(min_host_mem,available)
                if available < 1536 * 1024**2:
                    raise MemoryError("host memory below scored-tail safety floor")
            d=directions[start:start+direction_batch]
            width=d.shape[0]
            signed=torch.cat((d,-d),dim=0)
            jitter=signed[:,None,:].expand(
                2*width,count,out_features)*sigma
            losses,_=score_tail_scored(
                model,cache,a,b,jitter=jitter)
            plus,minus=losses[:width].float(),losses[width:].float()
            diff=plus-minus
            estimate += (
                diff.unsqueeze(-1)*d[:,None,:]
            ).sum(dim=0,keepdim=True)/(2.0*sigma*population)
            forwards += 1
        grad_a,grad_b=apply_local_lora_update_scored(
            cache,a,b,positions,estimate,count=count,lr=lr)
        return {
            "train_ce":float(clean.mean().item()),
            "tokens":count,
            "population":population,
            "direction_batch":direction_batch,
            "tail_forward_calls":forwards,
            "elapsed_seconds":time.monotonic()-started,
            "grad_a_norm":float(grad_a.norm().item()),
            "grad_b_norm":float(grad_b.norm().item()),
            "min_host_mem_available_bytes":min_host_mem,
            "scored_positions_only":True,
        }
