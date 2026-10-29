"""Offline, bounded *end-to-end graph* LFM3 layer-LoRA gradient smoke.

Applies a rank-4 additive adapter INSIDE the real final LFM3 FFN projection
during the full official-model forward pass. Loss gradients flow from the
actual vocabulary logits through later model layers into adapter weights.
Every pretrained parameter stays frozen. A true layer adapter, but only two
synthetic train and two synthetic dev prompts, one gradient step per arm.
Nothing is saved; no source traces, heldout, deployment or self-tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN, DEV, target_tokens, assert_fixed_lab_contract
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

TARGET = "model.layers.23.feed_forward.w2"
RANK = 4
LEARNING_RATE = 0.002
OPTIMIZERS = ("adamw", "muon_match_rms_adamw")
RESEARCH_SEEDS = (20261028, 20261029, 20261030)
MUON_SCALING_OPTIONS = ("muon_original", "muon_match_rms_adamw")


def install_adapter(layer, *, rank: int = RANK, seed: int = 20261028):
    """Attaches a temporary low-rank graph hook to one known 2-D torch.Linear."""
    import torch
    from torch import nn
    if (type(seed) is not int or seed not in RESEARCH_SEEDS
            or type(rank) is not int or rank != RANK
            or not isinstance(layer, nn.Linear)
            or layer.weight.ndim != 2
            or layer.weight.requires_grad
            or layer.in_features < 16 or layer.out_features < 16
            or layer.in_features > 32768 or layer.out_features > 32768):
        raise ResearchRefused("unknown or trainable LFM3 projection")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        A = nn.Linear(layer.in_features, rank, bias=False, dtype=torch.float32)
        B = nn.Linear(rank, layer.out_features, bias=False, dtype=torch.float32)
        nn.init.zeros_(B.weight)

    def hook(module, inputs, output):
        if (len(inputs) != 1 or not isinstance(output, torch.Tensor)
                or not isinstance(inputs[0], torch.Tensor)
                or inputs[0].shape[-1] != layer.in_features
                or output.shape[-1] != layer.out_features):
            raise ResearchRefused("unexpected LFM3 layer output while LoRA hooked")
        correction = B(A(inputs[0].float())) / rank
        return output + correction.to(dtype=output.dtype)

    handle = layer.register_forward_hook(hook)
    return A, B, handle


def run(snapshot: Path, *, steps: int = 1, multi_seed: bool = False,
        compare_muon_scaling: bool = False) -> dict:
    if (type(steps) is not int or steps not in (1, 3)
            or type(multi_seed) is not bool
            or type(compare_muon_scaling) is not bool
            or (multi_seed and steps != 3)
            or (compare_muon_scaling and not multi_seed)):
        raise ResearchRefused("only fixed three-seed, three-step research or approved smoke allowed")
    verify_restricted_environment()
    assert_fixed_lab_contract()
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("not the pinned isolated Transformers runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16
    )
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unexpected LFM3 architecture")
    model.eval()
    model.requires_grad_(False)
    base = tuple(model.parameters())
    before_versions = tuple(p._version for p in base)
    target = model.get_submodule(TARGET)
    if (not isinstance(target, torch.nn.Linear)
            or not target.weight.shape[0] == 2048
            or target.weight.shape[1] < 2048
            or target.weight.shape[1] > 32768):
        raise ResearchRefused("not the official final LFM3 feedforward projection")
    if not hasattr(torch.optim, "Muon"):
        raise ResearchRefused("native Torch Muon unavailable; no substitute")
    labels = target_tokens(tok)
    samples = []
    for rows in (TRAIN, DEV):
        split = []
        for prompt, letter in rows:
            encoded = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt"
            )
            if encoded["input_ids"].shape[1] > 80:
                raise ResearchRefused("synthetic train/development prompt too long")
            split.append((encoded, labels[letter]))
        samples.append(split)

    def loss_for(items):
        # End-to-end backpropagated from real model logits to the hooked LoRA;
        # no torch.no_grad scope during training, even though base is frozen.
        scores = []
        for encoded, token_id in items:
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
            scores.append(F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([token_id])
            ))
        return torch.stack(scores).mean()

    records = []
    for seed in (RESEARCH_SEEDS if multi_seed else RESEARCH_SEEDS[:1]):
        for optimizer_kind in (MUON_SCALING_OPTIONS if compare_muon_scaling else OPTIMIZERS):
            A, B, handle = install_adapter(target, seed=seed)
            trainables = list(A.parameters()) + list(B.parameters())
            if optimizer_kind == "adamw":
                opt = torch.optim.AdamW(trainables, lr=LEARNING_RATE, weight_decay=0.0)
            else:
                opt = torch.optim.Muon(
                    trainables, lr=LEARNING_RATE, momentum=0.95,
                    ns_steps=5, adjust_lr_fn=("original" if optimizer_kind == "muon_original"
                                              else "match_rms_adamw"), weight_decay=0.0
                )
            A_before = A.weight.detach().clone()
            try:
                with torch.no_grad():
                    before_train = float(loss_for(samples[0]))
                    before_dev = float(loss_for(samples[1]))
                max_grad_norm = 0.0
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    train_loss = loss_for(samples[0])
                    if not bool(torch.isfinite(train_loss)):
                        raise ResearchRefused("nonfinite end-to-end LoRA loss")
                    train_loss.backward()
                    grads = [p.grad for p in trainables]
                    if (grads[1] is None or not bool(torch.isfinite(grads[1]).all())
                            or float(grads[1].detach().abs().sum()) <= 0.0):
                        raise ResearchRefused("LoRA output matrix received no real-model gradient")
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
                    if not bool(torch.isfinite(grad_norm)):
                        raise ResearchRefused("nonfinite LoRA gradient")
                    max_grad_norm = max(max_grad_norm, float(grad_norm))
                    opt.step()
                    if (not all(bool(torch.isfinite(p).all()) for p in trainables)
                            or float(B.weight.detach().abs().sum()) <= 0.0):
                        raise ResearchRefused("LoRA matrices failed to update")
                if steps == 3 and bool(torch.equal(A.weight.detach(), A_before)):
                    raise ResearchRefused("three-step LoRA input matrix never updated")
                with torch.no_grad():
                    after_train = float(loss_for(samples[0]))
                    after_dev = float(loss_for(samples[1]))
                values = [before_train, before_dev, after_train, after_dev, float(grad_norm)]
                if not all(math.isfinite(x) for x in values):
                    raise ResearchRefused("nonfinite LoRA evaluation")
                records.append({
                    "optimizer": optimizer_kind,
                    **({"seed": seed} if multi_seed else {}),
                    "learning_rate": LEARNING_RATE,
                    "steps": steps,
                    "target": TARGET,
                    "rank": RANK,
                    "trainable_adapter_parameters": sum(p.numel() for p in trainables),
                    "train_loss_before": before_train,
                    "train_loss_after": after_train,
                    "dev_loss_before": before_dev,
                    "dev_loss_after": after_dev,
                    "gradient_norm_before_clipping": max_grad_norm,
                    "input_matrix_updated": not bool(torch.equal(A.weight.detach(), A_before)),
                    "verified_real_model_gradient": True,
                    "base_weight_update_executed": False,
                    "adapter_checkpoint_written": False,
                })
            finally:
                handle.remove()
    if (tuple(p._version for p in base) != before_versions
            or any(p.grad is not None or p.requires_grad for p in base)):
        raise ResearchRefused("frozen pretrained LFM3 was mutated")
    return {
        "schema": ("sg-lfm3-real-last-layer-muon-scaling-3seed.v1"
                   if compare_muon_scaling else
                   "sg-lfm3-real-last-layer-lora-3seed-3step.v1" if multi_seed
                   else "sg-lfm3-real-last-layer-lora-backprop-3step.v1"
                   if steps == 3 else "sg-lfm3-real-last-layer-lora-backprop.v1"),
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "model_revision": proof["revision"],
        "target_module": TARGET,
        "full_frozen_model_forward_used": True,
        "gradient_through_real_last_layer": True,
        "backbone_parameters_frozen": sum(p.numel() for p in base),
        "synthetic_train_sha256": hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "synthetic_dev_sha256": hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "two_train_two_dev_fabricated_only": True,
        "arms": records,
        "optimizer_model_chosen": False,
        "train_steps_per_arm": steps,
        **({"research_seeds": list(RESEARCH_SEEDS),
            "trial_count": len(records),
            **({ "per_seed_original_muon_wins": [
                    records[2*i]["dev_loss_after"] < records[2*i+1]["dev_loss_after"]
                    for i in range(len(RESEARCH_SEEDS))
                ]} if compare_muon_scaling else {
                    "per_seed_adamw_wins": [
                        records[2*i]["dev_loss_after"] < records[2*i+1]["dev_loss_after"]
                        for i in range(len(RESEARCH_SEEDS))
                    ]
                }),
            "eligible_for_manual_review_only": (not compare_muon_scaling and all(
                records[2*i]["dev_loss_after"] < records[2*i+1]["dev_loss_after"]
                for i in range(len(RESEARCH_SEEDS))
            ))} if multi_seed else {}),
        **({"muon_scaling_original_vs_rms": True,
            "muon_scaling_options": list(MUON_SCALING_OPTIONS),
            "may_nominate_optimizer_from_scaling_study": False}
           if compare_muon_scaling else {}),
        "heldout_evaluated": False,
        "source_custody_independently_verified": False,
        "base_model_weights_modified": False,
        "adapter_weights_saved": False,
        "may_promote_fleet": False,
        "production_ready": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", required=True, action="store_true")
    p.add_argument("--ack-sandbox-research", required=True, action="store_true")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--three-step-research", action="store_true")
    p.add_argument("--multi-seed-ffn-research", action="store_true")
    p.add_argument("--compare-muon-scaling-research", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(
        a.snapshot, steps=3 if a.three_step_research or a.multi_seed_ffn_research
        or a.compare_muon_scaling_research else 1,
        multi_seed=a.multi_seed_ffn_research or a.compare_muon_scaling_research,
        compare_muon_scaling=a.compare_muon_scaling_research), sort_keys=True))
