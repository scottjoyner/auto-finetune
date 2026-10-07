"""Benchmark cached final-o_proj tail replay against full K2 D=4 perturbations.

Research-only. Uses the frozen pretrained K2 base, deterministic train examples,
and temporary LoRA factors. No checkpoint, scheduler, deployment, or NAS writes.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from experiments.dust.k2_forward_only import FinalProjectionLoRA, position_losses
from experiments.dust.k2_gradient_calibration import select_train_example
from experiments.dust.k2_matched_compare import digest, load_base, read_mem_available
from experiments.dust.k2_structured_train_compare import (
    cosine_error, structured_step, structured_step_batched,
)
from experiments.dust.k2_tail_replay import (
    cache_base_sample, init_lora, tail_logits, tail_structured_step,
    tail_scored_structured_step,
)


def mean_ce(logits, labels) -> float:
    losses, mask = position_losses(logits, labels)
    return float((losses.sum() / mask.sum()).item())


def run(*, model_dir: Path, train_file: Path, expected_sha: str,
        device: str, seed: int, population: int, sigma: float, lr: float,
        direction_batch: int, steps: int, max_tokens: int,
        reference_mode: str, tail_mode: str) -> dict:
    import torch
    from transformers import AutoTokenizer

    if steps not in (1, 2, 4):
        raise ValueError("steps must be 1, 2 or 4")
    if direction_batch not in (2, 4, 8):
        raise ValueError("direction_batch must be 2, 4 or 8")
    if population not in (256, 512, 1024, 1536):
        raise ValueError("unsupported population")
    if direction_batch > population:
        raise ValueError("direction_batch exceeds population")
    if reference_mode not in ("serial", "batched"):
        raise ValueError("reference_mode must be serial or batched")
    if tail_mode not in ("full", "scored"):
        raise ValueError("tail_mode must be full or scored")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    samples = []
    selections = []
    for index in range(steps):
        sample, meta = select_train_example(
            train_file, tokenizer, index=index, max_tokens=max_tokens)
        samples.append(sample)
        selections.append(meta)
    del tokenizer
    gc.collect()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, model_sha = load_base(
        model_dir, device=device, expected_sha=expected_sha)
    original_weight = model.model.layers[-1].self_attn.o_proj.weight.detach().clone()

    try:
        cache_started = time.monotonic()
        caches = [cache_base_sample(model, sample, device) for sample in samples]
        if device == "cuda":
            torch.cuda.synchronize()
        cache_seconds = time.monotonic() - cache_started

        # Exact algebra check on the first real sample: base, nonzero LoRA, noise.
        first = samples[0]
        first_cache = caches[0]
        a_eq, b_eq = init_lora(first_cache, rank=4, seed=seed + 17)
        base_tail = tail_logits(model, first_cache, a_eq, b_eq)
        base_max_abs = float((base_tail - first_cache.base_logits).abs().max().item())
        base_ce_delta = mean_ce(base_tail, first_cache.labels) - mean_ce(
            first_cache.base_logits, first_cache.labels)

        generator = torch.Generator(device=device).manual_seed(seed + 9001)
        with torch.no_grad():
            b_eq.copy_(torch.randn(
                b_eq.shape, generator=generator, device=device) * .001)
        adapter = FinalProjectionLoRA(model, rank=4, seed=seed + 17)
        with torch.no_grad():
            adapter.a.copy_(a_eq)
            adapter.b.copy_(b_eq)
        ids = torch.tensor([first["tokens"]], device=device, dtype=torch.long)
        with torch.no_grad():
            full_lora = model(
                input_ids=ids, use_cache=False, return_dict=True).logits
        adapter.close()
        tail_lora = tail_logits(model, first_cache, a_eq, b_eq)
        lora_max_abs = float((tail_lora - full_lora).abs().max().item())
        lora_ce_delta = mean_ce(tail_lora, first_cache.labels) - mean_ce(
            full_lora, first_cache.labels)

        jitter = torch.randn(
            first_cache.oproj_output.shape, generator=generator,
            device=device, dtype=torch.float32) * sigma
        adapter = FinalProjectionLoRA(model, rank=4, seed=seed + 17)
        with torch.no_grad():
            adapter.a.copy_(a_eq)
            adapter.b.copy_(b_eq)
            adapter.jitter = jitter
            full_jitter = model(
                input_ids=ids, use_cache=False, return_dict=True).logits
        adapter.close()
        tail_jitter = tail_logits(
            model, first_cache, a_eq, b_eq, jitter=jitter)
        jitter_max_abs = float((tail_jitter - full_jitter).abs().max().item())
        jitter_ce_delta = mean_ce(
            tail_jitter, first_cache.labels) - mean_ce(
                full_jitter, first_cache.labels)

        # Full-model serial or D-batched structured reference path.
        full_adapter = FinalProjectionLoRA(model, rank=4, seed=seed + 17)
        full_a0 = full_adapter.a.detach().clone()
        full_b0 = full_adapter.b.detach().clone()
        full_history = []
        full_started = time.monotonic()
        for step, sample in enumerate(samples):
            if reference_mode == "serial":
                one = structured_step(
                    model, full_adapter, sample, device,
                    seed=seed * 100000 + step,
                    population=population, sigma=sigma, lr=lr)
            else:
                one = structured_step_batched(
                    model, full_adapter, sample, device,
                    seed=seed * 100000 + step,
                    population=population, sigma=sigma, lr=lr,
                    direction_batch=direction_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            full_history.append(one)
        full_seconds = time.monotonic() - full_started
        full_a = full_adapter.a.detach().clone() - full_a0
        full_b = full_adapter.b.detach().clone() - full_b0
        full_adapter.close()

        # Cached-tail path with identical initialization, order and directions.
        tail_a, tail_b = init_lora(
            caches[0], rank=4, seed=seed + 17)
        tail_a0 = tail_a.detach().clone()
        tail_b0 = tail_b.detach().clone()
        tail_history = []
        tail_started = time.monotonic()
        tail_step = (
            tail_scored_structured_step
            if tail_mode == "scored" else tail_structured_step)
        for step, cache in enumerate(caches):
            one = tail_step(
                model, cache, tail_a, tail_b,
                seed=seed * 100000 + step,
                population=population, sigma=sigma, lr=lr,
                direction_batch=direction_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            tail_history.append(one)
        tail_seconds = time.monotonic() - tail_started
        tail_a_update = tail_a.detach().clone() - tail_a0
        tail_b_update = tail_b.detach().clone() - tail_b0

        if not torch.equal(
                original_weight,
                model.model.layers[-1].self_attn.o_proj.weight):
            raise AssertionError("Frozen K2 o_proj weight changed")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Frozen base accumulated gradients")

        return {
            "schema": "auto-finetune.dust-k2-tail-replay-benchmark.v1",
            "research_only": True,
            "production_promotion_authorized": False,
            "model_weights_sha256": model_sha,
            "model_config_sha256": digest(model_dir / "config.json"),
            "seed": seed,
            "population": population,
            "sigma": sigma,
            "learning_rate": lr,
            "direction_batch": direction_batch,
            "steps": steps,
            "reference_mode": reference_mode,
            "tail_mode": tail_mode,
            "selections": selections,
            "equivalence": {
                "base_logits_max_abs": base_max_abs,
                "base_ce_delta": base_ce_delta,
                "lora_logits_max_abs": lora_max_abs,
                "lora_ce_delta": lora_ce_delta,
                "jitter_logits_max_abs": jitter_max_abs,
                "jitter_ce_delta": jitter_ce_delta,
            },
            "prefix_cache_seconds": cache_seconds,
            "full_model": {
                "reference_mode": reference_mode,
                "elapsed_seconds": full_seconds,
                "forward_calls": sum(x["forward_calls"] for x in full_history),
                "history": full_history,
            },
            "tail_replay": {
                "tail_mode": tail_mode,
                "elapsed_seconds": tail_seconds,
                "total_with_prefix_cache_seconds": tail_seconds + cache_seconds,
                "tail_forward_calls": sum(
                    x["tail_forward_calls"] for x in tail_history),
                "history": tail_history,
            },
            "update_alignment": {
                "a": cosine_error(tail_a_update, full_a),
                "b": cosine_error(tail_b_update, full_b),
            },
            "speedup_tail_only_vs_full": full_seconds / tail_seconds,
            "speedup_including_prefix_cache": (
                full_seconds / (tail_seconds + cache_seconds)),
            "max_device_memory_allocated": (
                torch.cuda.max_memory_allocated()
                if device == "cuda" else None),
            "remaining_host_mem_available_bytes": read_mem_available(),
            "base_unchanged": True,
            "base_model_grads_absent": True,
            "checkpoint_written": False,
            "raw_user_data_in_report": False,
            "warning": (
                "Final-layer o_proj research only; earlier layers and q/k/v "
                "require separate causal-credit design."),
        }
    finally:
        del model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--expected-sha256", default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--population", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=.25)
    ap.add_argument("--lr", type=float, default=.1)
    ap.add_argument("--direction-batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reference-mode", choices=("serial","batched"), default="serial")
    ap.add_argument("--tail-mode", choices=("full","scored"), default="scored")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    if args.output is not None and args.output.exists():
        ap.error("Refusing to overwrite existing evidence")
    report = run(
        model_dir=args.model_dir, train_file=args.train_jsonl,
        expected_sha=args.expected_sha256, device=args.device,
        seed=args.seed, population=args.population, sigma=args.sigma,
        lr=args.lr, direction_batch=args.direction_batch,
        steps=args.steps, max_tokens=args.max_tokens,
        reference_mode=args.reference_mode, tail_mode=args.tail_mode)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
