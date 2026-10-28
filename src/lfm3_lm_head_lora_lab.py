"""Actual LoRA on the official frozen LFM3 LM head, synthetic CPU lab.

Unlike the earlier hidden-state residual, this optimizes W_lm_head + B @ A
using a low-rank additive output projection. The full LFM3 backbone and
original LM-head W remain frozen; only A and B are trainable. Final hidden
states are extracted with NO gradient, so this is *head-only* LoRA on frozen
features, NOT end-to-end attention/FFN LoRA or full-model SFT.

The model and adapter weights are never saved. Train/dev prompts are fictional.
Muon on LoRA is a separately gated exploratory exception, NOT a default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import (
    TRAIN, DEV, assert_fixed_lab_contract, target_tokens,
)
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

RANK = 4
STEPS = 8
RATES = (0.001, 0.002)
TARGET = "model.lm_head.weight"


def fit_frozen_lm_head_lora(
    features, labels, frozen_head, *, learning_rate: float,
    optimizer: str = "adamw", muon_research_exception: bool = False,
    steps: int = STEPS, muon_lr_scaling: str = "original"
) -> dict:
    if (type(learning_rate) is not float or learning_rate not in RATES
            or type(steps) is not int or not 1 <= steps <= STEPS
            or optimizer not in ("adamw", "muon_research")
            or muon_lr_scaling not in ("original", "match_rms_adamw")
            or (optimizer == "adamw" and muon_lr_scaling != "original")
            or (optimizer == "muon_research" and muon_research_exception is not True)
            or (optimizer == "adamw" and muon_research_exception is not False)):
        raise ResearchRefused("LoRA optimizer admission/budget refused")
    import torch
    from torch import nn
    from torch.nn import functional as F

    (xt, xd), (yt, yd) = features, labels
    if (xt.shape != (2, 2048) or xd.shape != (2, 2048)
            or yt.shape != (2,) or yd.shape != (2,)
            or xt.dtype != torch.float32 or xd.dtype != torch.float32
            or not torch.isfinite(xt).all() or not torch.isfinite(xd).all()):
        raise ResearchRefused("frozen feature vectors differ from fixed lab contract")
    parameters = tuple(frozen_head.parameters())
    if (len(parameters) != 1 or parameters[0].ndim != 2
            or parameters[0].shape[1] != 2048
            or parameters[0].shape[0] < 32
            or parameters[0].requires_grad):
        raise ResearchRefused("LM-head original weight not frozen/recognized")
    vocab_size = int(parameters[0].shape[0])
    if int(yt.min()) < 0 or int(yd.min()) < 0 or max(
        int(yt.max()), int(yd.max())
    ) >= vocab_size:
        raise ResearchRefused("target token outside original fixed vocabulary")
    weight_version = parameters[0]._version
    with torch.no_grad():
        # Model's frozen vocabulary matrix is constant; cache the logits for
        # this tiny synthetic lab. B@A is added in float32, not into W.
        base_train = frozen_head(xt.to(torch.bfloat16)).float().detach()
        base_dev = frozen_head(xd.to(torch.bfloat16)).float().detach()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261028)
        A = nn.Linear(2048, RANK, bias=False, dtype=torch.float32)
        B = nn.Linear(RANK, vocab_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(B.weight)
    trainables = list(A.parameters()) + list(B.parameters())
    count = 2048 * RANK + vocab_size * RANK
    if sum(p.numel() for p in trainables) != count:
        raise ResearchRefused("LoRA parameter census invalid")
    if optimizer == "adamw":
        opt = torch.optim.AdamW(trainables, lr=learning_rate, weight_decay=0.0)
    else:
        if not hasattr(torch.optim, "Muon"):
            raise ResearchRefused("Muon unavailable; do not substitute AdamW")
        if tuple(A.weight.shape) != (RANK, 2048) or tuple(B.weight.shape) != (vocab_size, RANK):
            raise ResearchRefused("unreviewed Muon LoRA projection shape")
        opt = torch.optim.Muon(
            trainables, lr=learning_rate, momentum=0.95,
            ns_steps=5, weight_decay=0.0, adjust_lr_fn=muon_lr_scaling
        )
    scale = 1.0 / RANK

    def loss(x, base, labels):
        # Exact low-rank update to the LM-head output, with frozen head W.
        logits = base + scale * B(A(x))
        return F.cross_entropy(logits, labels)

    with torch.no_grad():
        before_train = float(loss(xt, base_train, yt))
        before_dev = float(loss(xd, base_dev, yd))
    peak_grad_norm = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        score = loss(xt, base_train, yt)
        if not bool(torch.isfinite(score)):
            raise ResearchRefused("nonfinite LoRA train loss")
        score.backward()
        norm = nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        if not bool(torch.isfinite(norm)):
            raise ResearchRefused("nonfinite LoRA gradient")
        peak_grad_norm = max(peak_grad_norm, float(norm))
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in trainables):
            raise ResearchRefused("nonfinite low rank update")
    with torch.no_grad():
        after_train = float(loss(xt, base_train, yt))
        after_dev = float(loss(xd, base_dev, yd))
        moved = any(bool(torch.count_nonzero(p).item()) for p in B.parameters())
    if (parameters[0]._version != weight_version or parameters[0].grad is not None
            or not moved or not all(math.isfinite(v) for v in
                (after_train, after_dev, peak_grad_norm))):
        raise ResearchRefused("frozen head integrity or adapter gradient check failed")
    return {
        "optimizer": ("adamw" if optimizer == "adamw" else (
            "muon_experimental" if muon_lr_scaling == "original"
            else "muon_experimental_match_rms_adamw"
        )),
        "learning_rate": learning_rate,
        "steps": steps,
        "target": TARGET,
        "lora_rank": RANK,
        "lora_alpha": 1.0,
        "vocab_size": vocab_size,
        "trainable_parameters": count,
        "train_loss_before": before_train,
        "train_loss_after": after_train,
        "dev_loss_before": before_dev,
        "dev_loss_after": after_dev,
        "peak_preclip_grad_norm": peak_grad_norm,
        "base_head_version_unchanged": True,
        "trained_adapter_written": False,
        "production_ready": False,
    }


def run(snapshot: Path, *, allow_muon: bool = False,
        compare_muon_lr_scaling: bool = False) -> dict:
    verify_restricted_environment()
    assert_fixed_lab_contract()
    if (type(allow_muon) is not bool
            or type(compare_muon_lr_scaling) is not bool
            or compare_muon_lr_scaling and not allow_muon):
        raise ResearchRefused("Muon scaling comparison needs explicit research exception")
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unexpected unpinned framework version")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16
    )
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unrecognized backbone class")
    model.eval()
    model.requires_grad_(False)
    weights = tuple(model.parameters())
    versions = tuple(x._version for x in weights)
    labels = target_tokens(tokenizer)
    split_features = []
    split_targets = []
    for rows in (TRAIN, DEV):
        hidden, target = [], []
        for prompt, letter in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt",
            )
            if ids["input_ids"].shape[1] > 80:
                raise ResearchRefused("synthetic source prompt over budget")
            with torch.inference_mode():
                out = model(**ids, output_hidden_states=True, use_cache=False)
                hidden.append(out.hidden_states[-1][0, -1, :].detach().clone().float())
                del out
            target.append(labels[letter])
        split_features.append(torch.stack(hidden))
        split_targets.append(torch.tensor(target, dtype=torch.long))
    arms = [
        fit_frozen_lm_head_lora(
            tuple(split_features), tuple(split_targets), model.lm_head,
            learning_rate=lr, optimizer="adamw",
        )
        for lr in RATES
    ]
    if allow_muon:
        arms.extend(fit_frozen_lm_head_lora(
            tuple(split_features), tuple(split_targets), model.lm_head,
            learning_rate=lr, optimizer="muon_research",
            muon_research_exception=True
        ) for lr in RATES)
    if compare_muon_lr_scaling:
        arms.extend(fit_frozen_lm_head_lora(
            tuple(split_features), tuple(split_targets), model.lm_head,
            learning_rate=lr, optimizer="muon_research",
            muon_research_exception=True,
            muon_lr_scaling="match_rms_adamw",
        ) for lr in RATES)
    if tuple(x._version for x in weights) != versions:
        raise ResearchRefused("base LFM3 parameters modified")
    if any(x.grad is not None for x in weights):
        raise ResearchRefused("gradient attached to pretrained weight")
    return {
        "schema": ("sg-lfm3-frozen-head-lora-muon-calibration.v1"
                   if compare_muon_lr_scaling
                   else "sg-lfm3-frozen-head-lora-lab.v1"),
        "source": "fixed-fabricated-two-train-two-dev-only",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "revision": proof["revision"],
        "lora_target": TARGET,
        "method": "exact-low-rank-update-to-frozen-lm-head-on-frozen-features",
        "end_to_end_backpropagated": False,
        "base_weights_modified": False,
        "training_split_sha256": hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "dev_split_sha256": hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "arms": arms,
        "explicit_muon_exception": allow_muon,
        "heldout_used": False,
        "source_custody_authenticated": False,
        "model_or_adapter_weights_written": False,
        "may_promote_to_fleet": False,
        "production_ready": False,
        **({"muon_rms_scaling_compare_enabled": True,
            "muon_rms_scaling_arm_indices": [4, 5],
            "muon_scaling_original_arm_indices": [2, 3]}
           if compare_muon_lr_scaling else {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", required=True, action="store_true")
    parser.add_argument("--ack-sandbox-research", required=True, action="store_true")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--experimental-muon-lora-exception", action="store_true")
    parser.add_argument("--compare-muon-rms-scaling", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.snapshot, allow_muon=args.experimental_muon_lora_exception,
        compare_muon_lr_scaling=args.compare_muon_rms_scaling
    ), sort_keys=True))
