"""Train/evaluate final-o_proj LoRA using cached K2 tail replay only.

Research-only optimization for the final decoder layer. The frozen prefix is
cached once per sample; orthogonal antithetic perturbations then run only over
supervised token positions through final MLP/norm/lm_head. No checkpoint write.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from experiments.dust.k2_data import select_disjoint
from experiments.dust.k2_matched_compare import digest, load_base, read_mem_available
from experiments.dust.k2_tail_replay import (
    cache_base_sample, init_lora, score_tail_scored,
    tail_scored_structured_step, tail_scored_tokenwise_step,
)


def score_caches(model, caches, a, b) -> dict:
    total=0.0
    tokens=0
    for cache in caches:
        losses,_=score_tail_scored(model,cache,a,b)
        total += float(losses.sum().item())
        tokens += int(losses.numel())
    mean=total/max(tokens,1)
    import math
    return {
        "tokens":tokens,
        "mean_token_ce":mean,
        "perplexity":math.exp(min(mean,80.0)),
    }


def comparable_dataset(current: dict, reference: dict) -> bool:
    keys=(
        "train_source","eval_source","train_selected_pair_sha256",
        "eval_selected_pair_sha256","train_rows","eval_rows",
        "train_max_seq_tokens","eval_max_seq_tokens",
    )
    return all(current[k] == reference[k] for k in keys)


def run(*, model_dir: Path, train_file: Path, heldout_file: Path,
        expected_sha: str, device: str, seed: int, population: int,
        sigma: float, lr: float, direction_batch: int, steps: int,
        train_count: int, eval_count: int, train_max_tokens: int,
        eval_max_tokens: int, reference_file: Path | None,
        estimator: str = "shared_orthogonal") -> dict:
    import torch
    from transformers import AutoTokenizer

    if not 1 <= steps <= train_count:
        raise ValueError("steps must be between 1 and train_count")
    if population not in (256,512,1024,1536):
        raise ValueError("unsupported population")
    if direction_batch not in (2,4,8,16):
        raise ValueError("unsupported direction_batch")
    if not (0 < sigma <= .5 and 0 < lr <= 1.0):
        raise ValueError("invalid sigma/lr")
    if estimator not in ("shared_orthogonal","tokenwise_gaussian"):
        raise ValueError("unsupported estimator")

    tok=AutoTokenizer.from_pretrained(
        str(model_dir),trust_remote_code=True,local_files_only=True)
    train,heldout,dataset=select_disjoint(
        train_file,heldout_file,tok,
        train_count=train_count,eval_count=eval_count,
        max_tokens=train_max_tokens,eval_max_tokens=eval_max_tokens)
    del tok
    gc.collect()

    reference=None
    if reference_file is not None:
        reference=json.loads(reference_file.read_text(encoding="utf-8"))
        if reference.get("schema")!="auto-finetune.dust-k2-structured-train.v1":
            raise ValueError("reference schema mismatch")
        if int(reference["seed"]) != seed:
            raise ValueError("reference seed mismatch")
        if not comparable_dataset(dataset,reference["dataset"]):
            raise ValueError("fresh data selection differs from immutable reference")

    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    model,model_sha=load_base(
        model_dir,device=device,expected_sha=expected_sha)
    original=model.model.layers[-1].self_attn.o_proj.weight.detach().clone()

    try:
        cache_started=time.monotonic()
        train_caches=[
            cache_base_sample(model,row,device,keep_logits=False)
            for row in train
        ]
        heldout_caches=[
            cache_base_sample(model,row,device,keep_logits=False)
            for row in heldout
        ]
        if device=="cuda":
            torch.cuda.synchronize()
        cache_seconds=time.monotonic()-cache_started

        a,b=init_lora(train_caches[0],rank=4,seed=seed+17)
        a0=a.detach().clone()
        b0=b.detach().clone()

        score_started=time.monotonic()
        train_before=score_caches(model,train_caches,a,b)
        held_before=score_caches(model,heldout_caches,a,b)
        pre_score_seconds=time.monotonic()-score_started

        history=[]
        train_started=time.monotonic()
        step_fn=(
            tail_scored_structured_step
            if estimator=="shared_orthogonal"
            else tail_scored_tokenwise_step)
        for step in range(steps):
            one=step_fn(
                model,train_caches[step],a,b,
                seed=seed*100000+step,population=population,
                sigma=sigma,lr=lr,direction_batch=direction_batch)
            if device=="cuda":
                torch.cuda.synchronize()
            history.append({"step":step+1,**one})
        train_seconds=time.monotonic()-train_started

        score_started=time.monotonic()
        train_after=score_caches(model,train_caches,a,b)
        held_after=score_caches(model,heldout_caches,a,b)
        post_score_seconds=time.monotonic()-score_started

        if not torch.equal(
            original,model.model.layers[-1].self_attn.o_proj.weight):
            raise AssertionError("frozen base o_proj changed")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("frozen base accumulated gradients")

        out={
            "schema":"auto-finetune.dust-k2-tail-train.v1",
            "research_only":True,
            "production_promotion_authorized":False,
            "model_weights_sha256":model_sha,
            "model_config_sha256":digest(model_dir/"config.json"),
            "seed":seed,"rank":4,"steps":steps,"population":population,
            "sigma":sigma,"learning_rate":lr,
            "direction_batch":direction_batch,
            "estimator":estimator,
            "independent_noise_per_token":estimator=="tokenwise_gaussian",
            "dataset":dataset,
            "train_before":train_before,"train_after":train_after,
            "heldout_before":held_before,"heldout_after":held_after,
            "train_ce_delta":(
                train_after["mean_token_ce"]-train_before["mean_token_ce"]),
            "heldout_ce_delta":(
                held_after["mean_token_ce"]-held_before["mean_token_ce"]),
            "history":history,
            "prefix_cache_seconds":cache_seconds,
            "pre_score_seconds":pre_score_seconds,
            "training_seconds":train_seconds,
            "post_score_seconds":post_score_seconds,
            "total_seconds":(
                cache_seconds+pre_score_seconds+train_seconds+post_score_seconds),
            "adapter_a_update_norm":float((a-a0).norm().item()),
            "adapter_b_update_norm":float((b-b0).norm().item()),
            "base_unchanged":True,
            "base_model_grads_absent":True,
            "backward_calls":0,
            "checkpoint_written":False,
            "raw_user_data_in_report":False,
            "max_device_memory_allocated":(
                torch.cuda.max_memory_allocated() if device=="cuda" else None),
            "remaining_host_mem_available_bytes":read_mem_available(),
            "warning":(
                "Final-layer tail-replay research only; quality/generalization "
                "and earlier-layer causal credit remain separate questions."),
        }
        if reference is not None:
            out["reference"]={
                "manifest_sha256":digest(reference_file),
                "serial_structured":{
                    "train_ce_delta":reference["structured"]["train_ce_delta"],
                    "heldout_ce_delta":reference["structured"]["heldout_ce_delta"],
                    "elapsed_seconds":reference["structured"]["elapsed_seconds"],
                },
                "backprop":{
                    "train_ce_delta":reference["backprop"]["train_ce_delta"],
                    "heldout_ce_delta":reference["backprop"]["heldout_ce_delta"],
                    "elapsed_seconds":reference["backprop"]["elapsed_seconds"],
                },
            }
        return out
    finally:
        del model


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--train-jsonl",type=Path,required=True)
    ap.add_argument("--heldout-jsonl",type=Path,required=True)
    ap.add_argument("--reference-json",type=Path)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--expected-sha256",default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--population",type=int,default=1024)
    ap.add_argument("--sigma",type=float,default=.25)
    ap.add_argument("--lr",type=float,default=.1)
    ap.add_argument("--direction-batch",type=int,default=4)
    ap.add_argument(
        "--estimator",choices=("shared_orthogonal","tokenwise_gaussian"),
        default="shared_orthogonal")
    ap.add_argument("--steps",type=int,default=4)
    ap.add_argument("--train-count",type=int,default=16)
    ap.add_argument("--eval-count",type=int,default=12)
    ap.add_argument("--train-max-tokens",type=int,default=128)
    ap.add_argument("--eval-max-tokens",type=int,default=512)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args(argv)
    if a.output is not None and a.output.exists():
        ap.error("Refusing to overwrite existing evidence")
    report=run(
        model_dir=a.model_dir,train_file=a.train_jsonl,
        heldout_file=a.heldout_jsonl,reference_file=a.reference_json,
        expected_sha=a.expected_sha256,device=a.device,seed=a.seed,
        population=a.population,sigma=a.sigma,lr=a.lr,
        direction_batch=a.direction_batch,steps=a.steps,
        estimator=a.estimator,
        train_count=a.train_count,eval_count=a.eval_count,
        train_max_tokens=a.train_max_tokens,
        eval_max_tokens=a.eval_max_tokens)
    rendered=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if a.output is None:
        print(rendered,end="")
    else:
        with a.output.open("x",encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
