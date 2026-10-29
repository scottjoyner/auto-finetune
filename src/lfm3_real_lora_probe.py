"""Opt-in real pretrained LFM3 LoRA experiment with synthetic tokens.

Never saves adapters, rewrites a checkpoint, or controls production services.
Requires the pinned official loader and an isolated offline CPU environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

MODEL_PATH = "/home/scott/lfm3-research/model-metadata"
BASE_SHA256 = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
SCHEMA = "sensorgraft.lfm3.real-lora-synthetic-probe.v1"
MAX_STEPS = 30
MAX_SEEDS = 3


def _sha(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _synthetic_token_data(torch, seed):
    g = torch.Generator(device="cpu").manual_seed(20261029 + seed)
    train = torch.randint(100, 4000, (16, 12), generator=g)
    heldout = torch.randint(100, 4000, (4, 12), generator=g)
    left = {row.numpy().tobytes() for row in train}
    right = {row.numpy().tobytes() for row in heldout}
    if left & right:
        raise RuntimeError("train_holdout_overlap")
    return train, heldout


def validate(*, seeds, steps, optimizers, adamw_lr, muon_lr, experimental_muon):
    if (
        not isinstance(seeds, (list, tuple))
        or not 1 <= len(seeds) <= MAX_SEEDS
        or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(steps) is not int
        or not 1 <= steps <= MAX_STEPS
        or not isinstance(optimizers, (list, tuple))
        or not optimizers
        or len(set(optimizers)) != len(optimizers)
        or any(name not in ("adamw", "muon") for name in optimizers)
        or type(experimental_muon) is not bool
        or type(adamw_lr) not in (float, int)
        or type(muon_lr) not in (float, int)
        or not math.isfinite(adamw_lr)
        or not math.isfinite(muon_lr)
        or not 1e-6 <= adamw_lr <= 2e-4
        or not 1e-5 <= muon_lr <= 2e-3
    ):
        raise ValueError("unbounded_or_invalid_probe_config")
    if "muon" in optimizers and not experimental_muon:
        raise ValueError("experimental_muon_lora_requires_explicit_opt_in")


def _preflight():
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or any(
            os.environ.get(k) != "-1"
            for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        )
    ):
        raise RuntimeError("offline_cpu_isolation_not_confirmed")


def _load(torch, seed, model_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.lfm3_pretrained_guard import (
        load_pretrained_lfm3,
        verify_official_lfm3_snapshot,
    )
    from src.lfm3_support import patch_transformers_for_lfm3

    verify_official_lfm3_snapshot(model_path)
    patch_transformers_for_lfm3()
    model = load_pretrained_lfm3(model_path, AutoConfig, AutoModelForCausalLM, torch.bfloat16)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("unexpected_gpu_or_meta_model")
    torch.manual_seed(91827 + seed)
    model.config.use_cache = False
    wrapper = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    named = [(n, p) for n, p in wrapper.named_parameters() if p.requires_grad]
    if (
        len(named) != 44
        or sum(p.numel() for _, p in named) != 146432
        or not all((".lora_A." in n or ".lora_B." in n) and p.ndim == 2 for n, p in named)
    ):
        raise RuntimeError("unexpected_trainable_lora_parameter_groups")
    if any(
        p.requires_grad
        for n, p in wrapper.named_parameters()
        if ".lora_A." not in n and ".lora_B." not in n
    ):
        raise RuntimeError("pretrained_backbone_not_frozen")
    return wrapper, named


def _eval(torch, model, heldout):
    model.eval()
    values = []
    with torch.inference_mode():
        for row in heldout:
            loss = float(
                model(input_ids=row.unsqueeze(0), labels=row.unsqueeze(0), use_cache=False).loss
            )
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite_heldout_loss")
            values.append(loss)
    return sum(values) / len(values)


def _arm(
    torch,
    *,
    seed,
    steps,
    optimizer_name,
    adamw_lr,
    muon_lr,
    train,
    heldout,
    train_sha,
    holdout_sha,
    model_path,
):
    model, named = _load(torch, seed, model_path)
    params = [p for _, p in named]
    first = _sha(*params)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=adamw_lr, weight_decay=0.0)
        lr = adamw_lr
    else:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch_muon_unavailable")
        optimizer = torch.optim.Muon(params, lr=muon_lr, weight_decay=0.0)
        lr = muon_lr
    backbone = next(
        p
        for n, p in model.named_parameters()
        if n.endswith("model.layers.2.self_attn.q_proj.base_layer.weight")
    )
    backbone_before = backbone.detach().reshape(-1)[:256].clone()
    before = _eval(torch, model, heldout)
    run_id = uuid.uuid4().hex
    events = []
    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        row = train[step % len(train)].unsqueeze(0)
        loss = model(input_ids=row, labels=row, use_cache=False).loss
        if not torch.isfinite(loss.detach()):
            raise RuntimeError("nonfinite_train_loss")
        loss.backward()
        grad_sq = sum(float(p.grad.detach().float().square().sum()) for p in params)
        if not math.isfinite(grad_sq):
            raise RuntimeError("nonfinite_gradient")
        snapshots = [p.detach().clone() for p in params]
        optimizer.step()
        update_sq = sum(
            float((p.detach() - old).float().square().sum()) for p, old in zip(params, snapshots)
        )
        if not math.isfinite(update_sq):
            raise RuntimeError("nonfinite_update")
        events.append(
            {
                "schema": "sensorgraft.lfm3.optimizer-step.v1",
                "run_id": run_id,
                "seed": seed,
                "step": step,
                "base_sha256": BASE_SHA256,
                "dataset_sha256": train_sha,
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "train_loss": float(loss.detach()),
                "grad_norm": math.sqrt(grad_sq),
                "update_norm": math.sqrt(update_sq),
                "elapsed_s": max(time.perf_counter() - t0, 1e-9),
            }
        )
    after = _eval(torch, model, heldout)
    if not torch.equal(backbone.detach().reshape(-1)[:256], backbone_before):
        raise RuntimeError("frozen_pretrained_backbone_mutated")
    result = {
        "seed": seed,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "steps": steps,
        "run_id": run_id,
        "base_sha256": BASE_SHA256,
        "initial_adapter_sha256": first,
        "final_adapter_sha256": _sha(*params),
        "train_sha256": train_sha,
        "heldout_sha256": holdout_sha,
        "heldout_loss_before": before,
        "heldout_loss_after": after,
        "relative_heldout_loss_change": after / before - 1,
        "frozen_backbone_sample_unchanged": True,
        "trainable_parameters": sum(p.numel() for p in params),
        "events": events,
        "weight_promotion_permitted": False,
    }
    if not math.isfinite(result["relative_heldout_loss_change"]):
        raise RuntimeError("invalid_evaluation_summary")
    del model, optimizer
    return result


def run_probe(
    *,
    seeds=(1,),
    steps=1,
    optimizers=("adamw",),
    adamw_lr=0.0001,
    muon_lr=0.0001,
    experimental_muon=False,
    model_path=MODEL_PATH,
):
    validate(
        seeds=seeds,
        steps=steps,
        optimizers=optimizers,
        adamw_lr=adamw_lr,
        muon_lr=muon_lr,
        experimental_muon=experimental_muon,
    )
    _preflight()
    import torch

    torch.set_num_threads(2)
    if "muon" in optimizers and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch_muon_unavailable")
    arms = []
    for seed in seeds:
        train, heldout = _synthetic_token_data(torch, seed)
        train_sha, holdout_sha = _sha(train), _sha(heldout)
        paired = []
        for name in optimizers:
            result = _arm(
                torch,
                seed=seed,
                steps=steps,
                optimizer_name=name,
                adamw_lr=adamw_lr,
                muon_lr=muon_lr,
                train=train,
                heldout=heldout,
                train_sha=train_sha,
                holdout_sha=holdout_sha,
                model_path=model_path,
            )
            paired.append(result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "optimizer": name,
                        "holdout_before": result["heldout_loss_before"],
                        "holdout_after": result["heldout_loss_after"],
                    }
                ),
                flush=True,
            )
        if len(paired) == 2 and any(
            paired[0][key] != paired[1][key]
            for key in (
                "base_sha256",
                "initial_adapter_sha256",
                "train_sha256",
                "heldout_sha256",
                "heldout_loss_before",
            )
        ):
            raise RuntimeError("optimizer_pair_starting_state_mismatch")
        arms.extend(paired)
    return {
        "schema": SCHEMA,
        "status": "research_synthetic_tokens_real_lfm3",
        "model": "LiquidAI/LFM3-1.7B",
        "base_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "base_sha256": BASE_SHA256,
        "real_pretrained_model": True,
        "synthetic_only": True,
        "real_private_traces_used": False,
        "heldout_used_for_optimizer_selection": True,
        "optimizer_parameter_policy": "experimental_muon_on_lora_factors"
        if "muon" in optimizers
        else "adamw_lora_only",
        "weight_promotion_permitted": False,
        "production_ready": False,
        "seeds": list(seeds),
        "steps": steps,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-only", action="store_true", required=True)
    parser.add_argument("--allow-real-lora-training", action="store_true", required=True)
    parser.add_argument("--allow-experimental-muon-lora", action="store_true")
    parser.add_argument("--optimizers", nargs="+", default=["adamw"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--adamw-lr", type=float, default=0.0001)
    parser.add_argument("--muon-lr", type=float, default=0.0001)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        seeds=args.seeds,
        steps=args.steps,
        optimizers=args.optimizers,
        adamw_lr=args.adamw_lr,
        muon_lr=args.muon_lr,
        experimental_muon=args.allow_experimental_muon_lora,
        model_path=str(args.model_path),
    )
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "weight_promotion_permitted": False,
            }
        )
    )


if __name__ == "__main__":
    main()