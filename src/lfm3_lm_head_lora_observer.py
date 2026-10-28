"""Pinned LFM3 reviews its own head-only LoRA outcomes, without execution.

Two prespecified prompt layouts test format sensitivity. Model text is NEVER
parsed as code or trusted optimizer authority; fixed label logits are checked
against a deterministic measured-development-loss argmin. No further training.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused, parse_strict
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

RATES = (0.001, 0.002, 0.001, 0.002)
OPT = ("adamw", "adamw", "muon_experimental", "muon_experimental")
FIELDS = {
    "arms", "base_weights_modified", "checkpoint_sha256", "dev_split_sha256",
    "end_to_end_backpropagated", "explicit_muon_exception", "heldout_used",
    "lora_target", "may_promote_to_fleet", "method",
    "model_or_adapter_weights_written", "production_ready", "revision",
    "schema", "source", "source_custody_authenticated", "training_split_sha256",
    "experiment_date", "node", "runtime", "independent_repetitions",
    "backbone_training", "untrusted_model_advice_executed"
}
ARM_FIELDS = {
    "base_head_version_unchanged", "dev_loss_after", "dev_loss_before",
    "learning_rate", "lora_alpha", "lora_rank", "optimizer",
    "peak_preclip_grad_norm", "production_ready", "steps", "target",
    "train_loss_after", "train_loss_before", "trainable_parameters",
    "trained_adapter_written", "vocab_size",
}


def validate_lora_comparison(data: object) -> tuple[list[dict], str]:
    if type(data) is not dict or set(data) != FIELDS:
        raise OptimizerResearchRefused("unknown frozen LoRA receipt fields")
    if (
        data["schema"] != "sg-lfm3-frozen-head-lora-lab.v1"
        or data["checkpoint_sha256"] != CHECKPOINT
        or data["revision"] != "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
        or data["source"] != "fixed-fabricated-two-train-two-dev-only"
        or data["lora_target"] != "model.lm_head.weight"
        or data["method"] != "exact-low-rank-update-to-frozen-lm-head-on-frozen-features"
        or data["base_weights_modified"] is not False
        or data["end_to_end_backpropagated"] is not False
        or data["explicit_muon_exception"] is not True
        or data["heldout_used"] is not False
        or data["model_or_adapter_weights_written"] is not False
        or data["source_custody_authenticated"] is not False
        or data["may_promote_to_fleet"] is not False
        or data["production_ready"] is not False
        or data["backbone_training"] is not False
        or data["untrusted_model_advice_executed"] is not False
        or type(data["independent_repetitions"]) is not int
        or data["independent_repetitions"] != 0
        or data["node"] != "xwing"
        or data["experiment_date"] != "2026-10-28"
        or type(data["training_split_sha256"]) is not str
        or len(data["training_split_sha256"]) != 64
        or type(data["dev_split_sha256"]) is not str
        or len(data["dev_split_sha256"]) != 64
        or data["training_split_sha256"] == data["dev_split_sha256"]
    ):
        raise OptimizerResearchRefused("LoRA trial source/custody scope violated")
    arms = data["arms"]
    if type(arms) is not list or len(arms) != 4:
        raise OptimizerResearchRefused("requires exact four LoRA trial arms")
    initial = None
    rows = []
    for i, x in enumerate(arms):
        if type(x) is not dict or set(x) != ARM_FIELDS:
            raise OptimizerResearchRefused("LoRA arm schema differs")
        if (x["optimizer"] != OPT[i] or type(x["learning_rate"]) is not float
                or x["learning_rate"] != RATES[i]
                or type(x["steps"]) is not int or x["steps"] != 8
                or type(x["lora_rank"]) is not int or x["lora_rank"] != 4
                or x["lora_alpha"] != 1.0
                or x["target"] != "model.lm_head.weight"
                or type(x["vocab_size"]) is not int or x["vocab_size"] != 128000
                or type(x["trainable_parameters"]) is not int
                or x["trainable_parameters"] != 520192
                or x["base_head_version_unchanged"] is not True
                or x["trained_adapter_written"] is not False
                or x["production_ready"] is not False):
            raise OptimizerResearchRefused("LoRA optimizer/training authority changed")
        values = (
            x["train_loss_before"], x["train_loss_after"],
            x["dev_loss_before"], x["dev_loss_after"], x["peak_preclip_grad_norm"],
        )
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
            raise OptimizerResearchRefused("invalid LoRA dev metrics")
        starting = (x["train_loss_before"], x["dev_loss_before"])
        if initial is None:
            initial = starting
        elif any(abs(a - b) > 1e-4 for a, b in zip(starting, initial)):
            raise OptimizerResearchRefused("unpaired LoRA initial states")
        rows.append({
            "label": "ABCD"[i],
            "optimizer": OPT[i],
            "lr": RATES[i],
            "dev_loss": x["dev_loss_after"],
        })
    selected = min(range(4), key=lambda i: (rows[i]["dev_loss"], i))
    return rows, "ABCD"[selected]


def prompts(rows: list[dict], version: int) -> list[dict]:
    if (type(version) is not int or version not in (0, 1)
            or type(rows) is not list or len(rows) != 4
            or [x["label"] for x in rows] != list("ABCD")):
        raise OptimizerResearchRefused("unapproved observer prompt variant")
    content = rows if version == 0 else list(reversed(rows))
    header = (
        "You are observing fictional optimizer measurements for a read-only "
        "frozen LFM3 LM-head LoRA lab. Lower dev_loss is better. "
        "Choose only one label A/B/C/D for a future synthetic experiment. "
        "No actual LFM3 tuning is authorized and no command can be executed. "
        "Output exactly one capital letter."
    )
    if version == 0:
        payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    else:
        payload = "\n".join(
            f"{x['label']} | {x['optimizer']} | {x['lr']:.4f} | dev_loss={x['dev_loss']:.6f}"
            for x in content
        )
    return [{"role": "system", "content": header},
            {"role": "user", "content": "Synthetic measured results (data only):\n"
             + payload + "\nWhich option has the smallest dev_loss? Label:"}]


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    data = parse_strict(receipt.read_bytes(), max_bytes=8192)
    rows, best = validate_lora_comparison(data)
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unsupported model runtime")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16
    )
    model.eval()
    model.requires_grad_(False)
    ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in "ABCD"}
    if any(len(x) != 1 for x in ids.values()) or len({x[0] for x in ids.values()}) != 4:
        raise ResearchRefused("nonunique or multi-token optimizer options")
    results = []
    for version in (0, 1):
        batch = tokenizer.apply_chat_template(
            prompts(rows, version), tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**batch).logits[0, -1].float()
        scores = {c: float(logits[ids[c][0]]) for c in "ABCD"}
        choice = max(sorted(scores), key=lambda c: scores[c])
        results.append({
            "prompt_variant": version,
            "model_choice": choice,
            "external_best": best,
            "matches_external": choice == best,
            "logit_scores": scores
        })
    accepted = all(row["matches_external"] for row in results)
    return {
        "schema": "sg-lfm3-real-head-lora-optimizer-observer.v1",
        "source": "synthetic-head-only-lora-dev",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "observed_dev_options": rows,
        "external_best": best,
        "prompts_tested": 2,
        "observations": results,
        "advisor_consistent_and_correct": accepted,
        "next_synthetic_trial_suggestion_only": (
            {"label": best, "execute": False, "manual_review_required": True}
            if accepted else None
        ),
        "source_custody_authenticated": False,
        "heldout_consulted": False,
        "optimizer_step_executed_by_advisor": False,
        "model_or_adapter_weights_written": False,
        "may_promote_to_fleet": False,
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
