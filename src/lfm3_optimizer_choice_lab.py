"""Logit-constrained LFM3 optimizer choice; no model-generated JSON executed.

The model scores six literal single-token option labels and the deterministic
policy expands the chosen label to an approved synthetic NEW-run proposal.
No training here, no heldout feedback, no source trace inputs, no direct writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_guard import (
    OptimizerResearchRefused, observer_view, parse_strict,
    validate_observation,
)
from src.lfm3_optimizer_advisor_lab import admission_from_model_advice
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

CHOICES = {
    "A": ("adamw", 1.0),
    "B": ("adamw", 0.5),
    "C": ("adamw", 2.0),
    "D": ("muon_hybrid", 1.0),
    "E": ("muon_hybrid", 0.5),
    "F": ("muon_hybrid", 2.0),
}


def admit_choice(letter: str, baseline: dict) -> dict:
    if type(letter) is not str or letter not in CHOICES:
        raise OptimizerResearchRefused("not a constrained optimizer option")
    name, mult = CHOICES[letter]
    return admission_from_model_advice(
        json.dumps({
            "target_optimizer": name, "lr_multiplier": mult,
            "reason_code": "controlled_optimizer_comparison",
        }, separators=(",", ":")).encode(),
        baseline,
    )


def choice_messages(view: dict) -> list[dict]:
    if (type(view) is not dict
        or view.get("schema") != "sg-lfm3-optimizer-observer-view.v1"
        or view.get("permitted_role") != "propose_new_synthetic_experiment_only"):
        raise OptimizerResearchRefused("unknown or overprivileged optimizer telemetry")
    return [
        {"role": "system", "content":
            "Review the following FABRICATED small-network optimizer dev results. "
            "Pick the safest ONE next experiment; no training execution here. "
            "Valid choices are A AdamW same LR, B AdamW half LR, C AdamW double LR, "
            "D Muon+AdamW same LR, E Muon+AdamW half LR, F Muon+AdamW double LR. "
            "Choose the single capital letter that best balances validation "
            "loss and stability. Output only A/B/C/D/E/F."},
        {"role": "user", "content":
            "Synthetic DEV-only telemetry (no actual LFM3 training or private traces): "
            + json.dumps(view, sort_keys=True, separators=(",", ":"))
            + "\nChoice: "},
    ]


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    report = parse_strict(receipt.read_bytes(), max_bytes=16384)
    if (type(report) is not dict
            or report.get("schema") != "sg-lfm3-optimizer-toy-paired.v1"
            or report.get("fabricated_only") is not True
            or report.get("trained_real_lfm3") is not False
            or report.get("production_ready") is not False):
        raise ResearchRefused("not a fabricated approved toy receipt")
    baseline = validate_observation(report["baseline"])
    view = observer_view(baseline)
    assets = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unverified isolated Transformers version")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("not the pinned architecture")
    codes = {}
    for code in CHOICES:
        token_ids = tok.encode(code, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ResearchRefused("constrained optimizer labels must be one token")
        codes[code] = token_ids[0]
    if len(set(codes.values())) != len(codes):
        raise ResearchRefused("optimizer option tokens must be distinct")
    batch = tok.apply_chat_template(
        choice_messages(view), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    with torch.inference_mode():
        logits = model(**batch).logits[0, -1].float()
    # Rank only the small fixed menu, never decode free-form model instructions.
    scores = {code: float(logits[tid]) for code, tid in codes.items()}
    choice = max(sorted(scores), key=lambda name: scores[name])
    proposal = admit_choice(choice, baseline)
    return {
        "schema": "sg-lfm3-optimizer-constrained-choice.v1",
        "source": "fabricated-toy-optimizer-metrics",
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "parent_observation_sha256": view["observation_sha256"],
        "options": list(CHOICES),
        "chosen_option": choice,
        "choice_logit_scores": scores,
        "admitted_new_trial_proposal": proposal,
        "freeform_model_json_executed": False,
        "model_checkpoint_modified": False,
        "optimizer_executed": False,
        "heldout_consulted": False,
        "production_ready": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--toy-receipt", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(run(a.snapshot, a.toy_receipt), sort_keys=True))
        return 0
    except (ValueError, OSError, ResearchRefused) as exc:
        raise SystemExit("constrained optimizer observer refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
