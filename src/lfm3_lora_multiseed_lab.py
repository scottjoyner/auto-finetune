"""Three-seed real frozen-LFM3 LM-head LoRA optimizer comparison.

Eighteen independent *new adapter* training runs use the exact same immutable
official frozen base, same fabricated 2-train/2-dev feature tensors, rank,
eight steps and optimizer budgets. This is only head-LoRA on frozen features,
not end-to-end LFM3 finetuning, full-model Muon, or recursive self-improvement.
Nothing is persisted beyond printed metric-only JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN, DEV, assert_fixed_lab_contract, target_tokens
from src.lfm3_lm_head_lora_lab import fit_frozen_lm_head_lora, RANK, STEPS
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

SEEDS = (20261028, 20261029, 20261030)
OPTIONS = (
    ("adamw", "original", 0.001),
    ("adamw", "original", 0.002),
    ("muon_research", "original", 0.001),
    ("muon_research", "original", 0.002),
    ("muon_research", "match_rms_adamw", 0.001),
    ("muon_research", "match_rms_adamw", 0.002),
)


def assess(rows: list[dict]) -> dict:
    if (type(rows) is not list or len(rows) != 3
            or [row.get("seed") for row in rows] != list(SEEDS)
            or any(type(row.get("arms")) is not list or len(row["arms"]) != 6 for row in rows)):
        raise ResearchRefused("missing fixed LoRA comparison trials")
    for row in rows:
        first = None
        for i, result in enumerate(row["arms"]):
            mode, scaling, rate = OPTIONS[i]
            expected = ("adamw" if mode == "adamw" else
                        "muon_experimental" if scaling == "original" else
                        "muon_experimental_match_rms_adamw")
            if (type(result) is not dict or result.get("optimizer") != expected
                    or result.get("seed") != row["seed"]
                    or result.get("learning_rate") != rate
                    or result.get("steps") != STEPS or result.get("lora_rank") != RANK
                    or result.get("trainable_parameters") != 520192
                    or result.get("target") != "model.lm_head.weight"
                    or result.get("trained_adapter_written") is not False
                    or result.get("base_head_version_unchanged") is not True
                    or result.get("production_ready") is not False):
                raise ResearchRefused("LoRA trial budget, seed or provenance changed")
            metrics = (
                result.get("train_loss_before"), result.get("train_loss_after"),
                result.get("dev_loss_before"), result.get("dev_loss_after"),
                result.get("peak_preclip_grad_norm"),
            )
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in metrics):
                raise ResearchRefused("invalid finite LoRA training/development metric")
            starting = (result["train_loss_before"], result["dev_loss_before"])
            if first is None:
                first = starting
            elif any(abs(a - b) > 1e-4 for a, b in zip(first, starting)):
                raise ResearchRefused("optimizer arms do not have identical base model predictions")
    medians = [statistics.median(row["arms"][i]["dev_loss_after"] for row in rows)
               for i in range(6)]
    best = min(range(6), key=lambda i: (medians[i], i))
    per_seed_best = [
        min(range(6), key=lambda i: (row["arms"][i]["dev_loss_after"], i))
        for row in rows
    ]
    muon_vs_adamw = [
        row["arms"][1]["dev_loss_after"] - row["arms"][5]["dev_loss_after"]
        for row in rows
    ]
    all_muon_improve = all(delta > 0 for delta in muon_vs_adamw)
    return {
        "schema": "sg-lfm3-frozen-lora-multiseed-assessment.v1",
        "three_seeds": list(SEEDS),
        "total_trials": 18,
        "median_dev_loss_by_arm": medians,
        "median_best_arm_index": best,
        "per_seed_best_arm_index": per_seed_best,
        "rms_muon_dev_improvement_vs_adamw_lr002_by_seed": muon_vs_adamw,
        "rms_muon_better_in_every_seed": all_muon_improve,
        "may_nominate_new_synthetic_research_trial_for_review":
            best == 5 and all_muon_improve,
        "statistically_confirmed": False,
        "heldout_evaluated": False,
        "source_authenticated": False,
        "automatic_optimizer_update_allowed": False,
        "production_ready": False,
    }


def run(snapshot: Path) -> dict:
    verify_restricted_environment()
    assert_fixed_lab_contract()
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("not the pinned LFM3 research runtime")
    if not hasattr(torch.optim, "Muon"):
        raise ResearchRefused("native Muon unavailable; do not substitute")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16)
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unexpected model architecture")
    model.eval()
    model.requires_grad_(False)
    base = tuple(model.parameters())
    before_versions = tuple(p._version for p in base)
    classes = target_tokens(tokenizer)
    features, labels = [], []
    for split in (TRAIN, DEV):
        hidden, targets = [], []
        for prompt, target in split:
            batch = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt")
            if batch["input_ids"].shape[1] > 80:
                raise ResearchRefused("fictional prompt exceeds hard context limit")
            with torch.inference_mode():
                out = model(**batch, output_hidden_states=True, use_cache=False)
                hidden.append(out.hidden_states[-1][0, -1].detach().clone().float())
            targets.append(classes[target])
        features.append(torch.stack(hidden))
        labels.append(torch.tensor(targets, dtype=torch.long))
    rows = []
    for seed in SEEDS:
        arms = []
        for mode, scaling, rate in OPTIONS:
            arms.append(fit_frozen_lm_head_lora(
                tuple(features), tuple(labels), model.lm_head,
                learning_rate=rate, optimizer=mode,
                muon_research_exception=(mode == "muon_research"),
                muon_lr_scaling=scaling,
                seed=seed,
            ))
        rows.append({"seed": seed, "arms": arms})
    if (tuple(p._version for p in base) != before_versions
            or any(p.grad is not None or p.requires_grad for p in base)):
        raise ResearchRefused("pretrained LFM3 tensors were modified")
    return {
        "schema": "sg-lfm3-frozen-head-lora-multiseed.v1",
        "model": "pinned_official_lfm3_bf16",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "model_revision": proof["revision"],
        "fictional_train_sha256": hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "fictional_dev_sha256": hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "source": "two-fabricated-train-two-fabricated-dev",
        "option_order": [{"optimizer": a, "scale": b, "learning_rate": c}
                         for a, b, c in OPTIONS],
        "seeds": list(SEEDS),
        "trials": rows,
        "assessment": assess(rows),
        "frozen_pretrained_parameter_count": sum(x.numel() for x in base),
        "backbone_or_head_weights_modified": False,
        "optimizer_state_written": False,
        "trained_adapter_weights_written": False,
        "model_was_finetuned_end_to_end": False,
        "used_real_source_traces": False,
        "heldout_test_used": False,
        "may_promote_fleet": False,
        "production_ready": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", required=True, action="store_true")
    p.add_argument("--ack-sandbox-research", required=True, action="store_true")
    p.add_argument("--snapshot", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.snapshot), sort_keys=True))
