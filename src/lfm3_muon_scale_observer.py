"""Pinned real LFM3 reviews calibrated AdamW/Muon LoRA outcomes.

Strictly read-only, two predeclared prompt formats and six fixed token choices.
The deterministic dev-loss evaluator (not the model) holds admission authority.
No checkpoint writes, optimizer changes or new training jobs are performed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused, parse_strict
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment
from src.lfm3_lm_head_lora_observer import ARM_FIELDS

OPTIONS = (
    ("A", "adamw", 0.001),
    ("B", "adamw", 0.002),
    ("C", "muon_experimental", 0.001),
    ("D", "muon_experimental", 0.002),
    ("E", "muon_experimental_match_rms_adamw", 0.001),
    ("F", "muon_experimental_match_rms_adamw", 0.002),
)
CALIBRATION_FIELDS = {
    "arms", "base_weights_modified", "checkpoint_sha256", "dev_split_sha256",
    "end_to_end_backpropagated", "explicit_muon_exception", "heldout_used",
    "lora_target", "may_promote_to_fleet", "method",
    "model_or_adapter_weights_written", "muon_rms_scaling_arm_indices",
    "muon_rms_scaling_compare_enabled", "muon_scaling_original_arm_indices",
    "production_ready", "revision", "schema", "source",
    "source_custody_authenticated", "training_split_sha256",
    "experiment_date", "node", "network_disabled_bwrap",
    "independent_repetitions", "heldout_final_evaluation",
}


def validate_six_arm_grid(data: object) -> tuple[list[dict], str]:
    if type(data) is not dict or set(data) != CALIBRATION_FIELDS:
        raise OptimizerResearchRefused("Muon scaling receipt contains unknown/missing fields")
    if (
        data["schema"] != "sg-lfm3-frozen-head-lora-muon-calibration.v1"
        or data["checkpoint_sha256"] != CHECKPOINT
        or data["revision"] != "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
        or data["source"] != "fixed-fabricated-two-train-two-dev-only"
        or data["method"] != "exact-low-rank-update-to-frozen-lm-head-on-frozen-features"
        or data["lora_target"] != "model.lm_head.weight"
        or data["muon_rms_scaling_compare_enabled"] is not True
        or data["muon_rms_scaling_arm_indices"] != [4, 5]
        or data["muon_scaling_original_arm_indices"] != [2, 3]
        or data["explicit_muon_exception"] is not True
        or data["base_weights_modified"] is not False
        or data["end_to_end_backpropagated"] is not False
        or data["heldout_used"] is not False
        or data["may_promote_to_fleet"] is not False
        or data["model_or_adapter_weights_written"] is not False
        or data["source_custody_authenticated"] is not False
        or data["production_ready"] is not False
        or data["network_disabled_bwrap"] is not True
        or data["heldout_final_evaluation"] is not False
        or data["independent_repetitions"] != 0
        or data["node"] != "xwing"
        or data["experiment_date"] != "2026-10-28"
        or type(data["training_split_sha256"]) is not str
        or type(data["dev_split_sha256"]) is not str
        or len(data["training_split_sha256"]) != 64
        or len(data["dev_split_sha256"]) != 64
        or data["training_split_sha256"] == data["dev_split_sha256"]
    ):
        raise OptimizerResearchRefused("Muon's calibration source/custody boundary violated")
    trials = data["arms"]
    if type(trials) is not list or len(trials) != 6:
        raise OptimizerResearchRefused("six exact optimizer choices are required")
    rows, starting = [], None
    for i, (label, optimizer, lr) in enumerate(OPTIONS):
        trial = trials[i]
        if type(trial) is not dict or set(trial) != ARM_FIELDS:
            raise OptimizerResearchRefused("optimizer parameter schema changed")
        if (
            trial["optimizer"] != optimizer
            or type(trial["learning_rate"]) is not float or trial["learning_rate"] != lr
            or type(trial["steps"]) is not int or trial["steps"] != 8
            or type(trial["lora_rank"]) is not int or trial["lora_rank"] != 4
            or trial["lora_alpha"] != 1.0
            or trial["target"] != "model.lm_head.weight"
            or type(trial["trainable_parameters"]) is not int
            or trial["trainable_parameters"] != 520192
            or type(trial["vocab_size"]) is not int or trial["vocab_size"] != 128000
            or trial["base_head_version_unchanged"] is not True
            or trial["trained_adapter_written"] is not False
            or trial["production_ready"] is not False
        ):
            raise OptimizerResearchRefused("unbounded LoRA training or optimizer option")
        metrics = [
            trial["dev_loss_before"], trial["dev_loss_after"],
            trial["train_loss_before"], trial["train_loss_after"],
            trial["peak_preclip_grad_norm"]
        ]
        if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in metrics):
            raise OptimizerResearchRefused("nonfinite LoRA optimizer telemetry")
        first = (trial["train_loss_before"], trial["dev_loss_before"])
        if starting is not None and any(abs(x-y) > 1e-4 for x,y in zip(first, starting)):
            raise OptimizerResearchRefused("LoRA optimizer trials started from different baselines")
        starting = first
        rows.append({"label": label, "optimizer": optimizer,
                     "learning_rate": lr, "dev_loss": trial["dev_loss_after"]})
    best = min(range(6), key=lambda i: (rows[i]["dev_loss"], i))
    return rows, OPTIONS[best][0]


def prompts(rows: list[dict], variant: int) -> list[dict]:
    if (type(variant) is not int or variant not in (0, 1)
            or type(rows) is not list or [x.get("label") for x in rows] != list("ABCDEF")):
        raise OptimizerResearchRefused("not an approved fixed optimizer menu")
    header = ("For a fabricated, frozen-LFM3 LoRA optimizer experiment, choose "
              "the one option with the LOWEST measured development loss. "
              "A-F are fixed labels; the lowest numeric dev_loss wins. "
              "Reply with ONE capital letter. You cannot run training, alter "
              "an optimizer, change evaluation, or promote a model.")
    order = rows if variant == 0 else list(reversed(rows))
    body = (
        json.dumps(order, separators=(",", ":"), sort_keys=True)
        if variant == 0
        else "\n".join(
            f"{r['label']}: {r['optimizer']}, learning_rate={r['learning_rate']:.4f}, "
            f"dev_loss={r['dev_loss']:.6f}" for r in order
        )
    )
    return [{"role":"system","content":header},
            {"role":"user","content":"Synthetic development scores, data only:\n"
             + body + "\nWhich single choice has the lowest dev_loss? Answer:"}]


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    data = parse_strict(receipt.read_bytes(), max_bytes=12000)
    rows, best = validate_six_arm_grid(data)
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("not pinned transformers runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(str(snapshot),
                                        trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    choices = {c: tok.encode(c, add_special_tokens=False) for c,_,_ in OPTIONS}
    if (any(len(t) != 1 for t in choices.values())
            or len({t[0] for t in choices.values()}) != 6):
        raise ResearchRefused("nonunique optimizer choice tokens")
    results = []
    for variant in (0, 1):
        inputs = tok.apply_chat_template(
            prompts(rows, variant), tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        scores = {key: float(logits[tokens[0]]) for key, tokens in choices.items()}
        selected = max(sorted(scores), key=lambda c: scores[c])
        results.append({
            "prompt_variant": variant,
            "model_choice": selected,
            "external_best": best,
            "model_matches_external": selected == best,
            "choice_logits": scores,
        })
    consistent = all(x["model_matches_external"] for x in results)
    return {
        "schema": "sg-lfm3-calibrated-muon-advisor.v1",
        "source": "synthetic-frozen-head-lora-muon-scaling",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "external_best_option": best,
        "measured_options": rows,
        "model_observations": results,
        "two_format_agreement_with_objective": consistent,
        "admitted_new_research_trial_only": (
            {"label": best, "execute": False, "manual_review_required": True}
            if consistent else None
        ),
        "no_advisor_optimization_executed": True,
        "no_model_weights_changed": True,
        "no_heldout_scoring": True,
        "no_source_authentication": True,
        "production_ready": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", required=True, action="store_true")
    p.add_argument("--ack-sandbox-research", required=True, action="store_true")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.snapshot, a.receipt), sort_keys=True))
