
"""Calibrate shared-direction vs independent-token activation perturbations.

Research-only comparison for the frozen K2-Horizon final-layer o_proj LoRA.
The exact local gradient is computed with autograd only as a calibration
reference. Both zeroth-order estimators themselves use forward evaluations
only and do not mutate a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
import time


def parse_ints(value: str) -> list[int]:
    out=[int(x) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def parse_floats(value: str) -> list[float]:
    out=[float(x) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return out


def run(*, model_dir: Path, train_file: Path, expected_sha: str, device: str,
        sample_index: int, max_tokens: int, adapter_seed: int, b_scale: float,
        seeds: list[int], populations: list[int], sigmas: list[float],
        direction_batch: int, max_seconds: int) -> dict:
    import torch
    from transformers import AutoTokenizer

    from experiments.dust.k2_gradient_calibration import select_train_example
    from experiments.dust.k2_matched_compare import load_base, digest, read_mem_available
    from experiments.dust.k2_structured_train_compare import cosine_error
    from experiments.dust.k2_tail_replay import (
        cache_base_sample, init_lora, local_lora_grads_scored,
        score_tail_scored, supervised_source_positions,
        tail_scored_structured_estimate, tail_scored_tokenwise_estimate,
    )

    if any(k <= 0 or k > 1536 for k in populations):
        raise ValueError("parity populations must be in [1,1536]")
    if any(s <= 0 or s > .5 for s in sigmas):
        raise ValueError("sigmas must be in (0,.5]")
    if direction_batch not in (1,2,4,8,16,32):
        raise ValueError("unsupported direction batch")
    if not 0 < b_scale <= .1:
        raise ValueError("b_scale must be in (0,.1]")
    if max_seconds < 30 or max_seconds > 1800:
        raise ValueError("max_seconds outside [30,1800]")

    tokenizer=AutoTokenizer.from_pretrained(
        str(model_dir),trust_remote_code=True,local_files_only=True)
    sample,selection=select_train_example(
        train_file,tokenizer,index=sample_index,max_tokens=max_tokens)
    del tokenizer

    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    model,model_sha=load_base(
        model_dir,device=device,expected_sha=expected_sha)
    original=model.model.layers[-1].self_attn.o_proj.weight.detach().clone()
    started=time.monotonic()
    try:
        cache=cache_base_sample(model,sample,device,keep_logits=False)
        a,b=init_lora(cache,rank=4,seed=adapter_seed)
        generator=torch.Generator(device=device).manual_seed(adapter_seed+1)
        with torch.no_grad():
            b.copy_(torch.randn(
                b.shape,generator=generator,device=device,dtype=torch.float32)
                * b_scale)

        positions,_=supervised_source_positions(cache)
        count=int(positions.numel())
        jitter=torch.zeros(
            1,count,cache.oproj_output.shape[-1],
            device=device,dtype=torch.float32,requires_grad=True)
        losses,_=score_tail_scored(model,cache,a,b,jitter=jitter)
        exact_output=torch.autograd.grad(losses.sum(),jitter)[0].detach()
        exact_a,exact_b=local_lora_grads_scored(
            cache,a,b,positions,exact_output,count=count)
        del jitter,losses

        points=[]
        for seed in seeds:
            for sigma in sigmas:
                for population in populations:
                    if time.monotonic()-started > max_seconds:
                        raise TimeoutError("tokenwise parity calibration exceeded wall budget")
                    shared=tail_scored_structured_estimate(
                        model,cache,a,b,seed=seed,population=population,
                        sigma=sigma,direction_batch=direction_batch)
                    tokenwise=tail_scored_tokenwise_estimate(
                        model,cache,a,b,seed=seed,population=population,
                        sigma=sigma,direction_batch=direction_batch)
                    for name,result in (
                        ("shared_orthogonal_antithetic",shared),
                        ("tokenwise_gaussian_antithetic",tokenwise),
                    ):
                        est_a,est_b=local_lora_grads_scored(
                            cache,a,b,result["positions"],result["estimate"],
                            count=count)
                        points.append({
                            "estimator":name,
                            "seed":seed,
                            "sigma":sigma,
                            "population":population,
                            "output_gradient":cosine_error(
                                result["estimate"],exact_output),
                            "lora_a_gradient":cosine_error(est_a,exact_a),
                            "lora_b_gradient":cosine_error(est_b,exact_b),
                            "elapsed_seconds":result["elapsed_seconds"],
                            "tail_forward_calls":result["tail_forward_calls"],
                            "unique_direction_vectors":result[
                                "unique_direction_vectors"],
                            "token_position_perturbations":result[
                                "token_position_perturbations"],
                            "independent_noise_per_token":result[
                                "independent_noise_per_token"],
                            "min_host_mem_available_bytes":result[
                                "min_host_mem_available_bytes"],
                        })

        if not torch.equal(
            original,model.model.layers[-1].self_attn.o_proj.weight):
            raise AssertionError("frozen base o_proj changed")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("frozen base accumulated gradients")

        summary=[]
        for estimator in sorted({p["estimator"] for p in points}):
            for sigma in sorted({p["sigma"] for p in points}):
                for population in sorted({p["population"] for p in points}):
                    rows=[
                        p for p in points
                        if p["estimator"]==estimator
                        and p["sigma"]==sigma
                        and p["population"]==population
                    ]
                    if not rows:
                        continue
                    def stat(path):
                        values=[r[path]["cosine"] for r in rows]
                        values=[v for v in values if v is not None]
                        return {
                            "mean":mean(values),
                            "pstdev":pstdev(values) if len(values)>1 else 0.0,
                            "min":min(values),
                            "max":max(values),
                        }
                    summary.append({
                        "estimator":estimator,
                        "sigma":sigma,
                        "population":population,
                        "seed_count":len(rows),
                        "output_gradient_cosine":stat("output_gradient"),
                        "lora_a_gradient_cosine":stat("lora_a_gradient"),
                        "lora_b_gradient_cosine":stat("lora_b_gradient"),
                        "mean_elapsed_seconds":mean(
                            r["elapsed_seconds"] for r in rows),
                        "mean_unique_direction_vectors":mean(
                            r["unique_direction_vectors"] for r in rows),
                        "mean_token_position_perturbations":mean(
                            r["token_position_perturbations"] for r in rows),
                    })

        return {
            "schema":"auto-finetune.dust-k2-tokenwise-parity.v1",
            "research_only":True,
            "production_promotion_authorized":False,
            "model_weights_sha256":model_sha,
            "model_config_sha256":digest(model_dir/"config.json"),
            "selection":selection,
            "sample_index":sample_index,
            "max_tokens":max_tokens,
            "scored_tokens":count,
            "rank":4,
            "adapter_seed":adapter_seed,
            "calibration_b_scale":b_scale,
            "seeds":seeds,
            "populations":populations,
            "sigmas":sigmas,
            "direction_batch":direction_batch,
            "exact_output_gradient_norm":float(exact_output.norm().item()),
            "exact_lora_a_gradient_norm":float(exact_a.norm().item()),
            "exact_lora_b_gradient_norm":float(exact_b.norm().item()),
            "points":points,
            "summary":summary,
            "elapsed_seconds":time.monotonic()-started,
            "max_device_memory_allocated":(
                torch.cuda.max_memory_allocated() if device=="cuda" else None),
            "remaining_host_mem_available_bytes":read_mem_available(),
            "base_unchanged":True,
            "base_model_grads_absent":True,
            "checkpoint_written":False,
            "raw_user_data_in_report":False,
            "warning":(
                "Final-o_proj local calibration only. The tokenwise estimator "
                "matches independent per-token noise semantics more closely, "
                "but this is not a full reproduction of Dust attention credit."),
        }
    finally:
        del model


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--train-jsonl",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--expected-sha256",default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    ap.add_argument("--sample-index",type=int,default=0)
    ap.add_argument("--max-tokens",type=int,default=128)
    ap.add_argument("--adapter-seed",type=int,default=59)
    ap.add_argument("--b-scale",type=float,default=.001)
    ap.add_argument("--seeds",type=parse_ints,default=[7,42,1337])
    ap.add_argument("--populations",type=parse_ints,default=[64,256,1024])
    ap.add_argument("--sigmas",type=parse_floats,default=[.05,.25])
    ap.add_argument("--direction-batch",type=int,default=4)
    ap.add_argument("--max-seconds",type=int,default=900)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args(argv)
    if args.output is not None and args.output.exists():
        ap.error("Refusing to overwrite existing evidence")
    report=run(
        model_dir=args.model_dir,train_file=args.train_jsonl,
        expected_sha=args.expected_sha256,device=args.device,
        sample_index=args.sample_index,max_tokens=args.max_tokens,
        adapter_seed=args.adapter_seed,b_scale=args.b_scale,
        seeds=args.seeds,populations=args.populations,sigmas=args.sigmas,
        direction_batch=args.direction_batch,max_seconds=args.max_seconds)
    rendered=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output is None:
        print(rendered,end="")
    else:
        with args.output.open("x",encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
