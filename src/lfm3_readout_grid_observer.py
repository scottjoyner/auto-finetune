"""Read-only LFM3 observes its own frozen-readout AdamW-vs-Muon test.

Four fixed choices, externally scored. There is NO optimizer execution,
raw trace input, model weight update or permission to self-promote.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from src.lfm3_frozen_readout_observer import (
    validate_readout_receipt, RECEIPT_FIELDS, ARM_FIELDS,
)
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_guard import parse_strict, OptimizerResearchRefused
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

OPTIONS = (
    ("A", "adamw_torch", 0.001),
    ("B", "adamw_torch", 0.002),
    ("C", "muon_torch_on_toy_readout_only", 0.001),
    ("D", "muon_torch_on_toy_readout_only", 0.002),
)
EXTRA = {"muon_research_exception_only_for_two_readout_matrices",
         "actual_full_lfm3_finetuning"}
ARM_EXTRA = {"muon_on_actual_lfm3_weights"}


def readout_grid_view(data: object) -> tuple[dict, str]:
    if (type(data) is not dict
            or set(data) != RECEIPT_FIELDS | EXTRA
            or data["schema"] != "sg-lfm3-frozen-readout-muon-grid.v1"
            or data["muon_research_exception_only_for_two_readout_matrices"] is not True
            or data["actual_full_lfm3_finetuning"] is not False
            or data["no_muon_adapter_optimization_executed"] is not False):
        raise OptimizerResearchRefused("not an explicitly reviewed toy Muon grid")
    arms = data["arms"]
    if type(arms) is not list or len(arms) != 4:
        raise OptimizerResearchRefused("four fixed readout optimizer arms required")
    # Pass the real AdamW pair through the existing pinned research validator.
    adamw_part = dict(data)
    for key in EXTRA:
        adamw_part.pop(key)
    adamw_part["schema"] = "sg-lfm3-frozen-readout-adapter.v1"
    adamw_part["no_muon_adapter_optimization_executed"] = True
    adamw_part["arms"] = []
    for arm in arms[:2]:
        if type(arm) is not dict or set(arm) != ARM_FIELDS | ARM_EXTRA:
            raise OptimizerResearchRefused("invalid AdamW readout arm")
        if arm["muon_on_actual_lfm3_weights"] is not False:
            raise OptimizerResearchRefused("pretrained weights touched")
        one = dict(arm)
        one.pop("muon_on_actual_lfm3_weights")
        adamw_part["arms"].append(one)
    validate_readout_receipt(adamw_part)

    before_train = arms[0]["train_loss_before"]
    before_dev = arms[0]["dev_loss_before"]
    for index, arm in enumerate(arms):
        code, optimizer, lr = OPTIONS[index]
        if type(arm) is not dict or set(arm) != ARM_FIELDS | ARM_EXTRA:
            raise OptimizerResearchRefused("readout grid fields inconsistent")
        if (arm["optimizer"] != optimizer
                or type(arm["learning_rate"]) is not float
                or arm["learning_rate"] != lr
                or type(arm["steps"]) is not int or arm["steps"] != 6
                or type(arm["trainable_residual_parameters"]) is not int
                or arm["trainable_residual_parameters"] != 16384
                or arm["rank"] != 4
                or arm["weights_saved"] is not False
                or arm["production_ready"] is not False
                or arm["muon_on_actual_lfm3_weights"] is not False):
            raise OptimizerResearchRefused("unapproved optimizer state or training budget")
        for k in ("train_loss_before", "dev_loss_before", "train_loss_after",
                  "dev_loss_after", "max_preclip_grad_norm"):
            if (type(arm[k]) not in (float, int)
                    or not math.isfinite(arm[k]) or arm[k] < 0):
                raise OptimizerResearchRefused("nonfinite or invalid readout observation")
        if (abs(arm["train_loss_before"] - before_train) > 1e-4
                or abs(arm["dev_loss_before"] - before_dev) > 1e-4):
            raise OptimizerResearchRefused("optimizer not paired from identical frozen features")
    view = {
        "schema": "sg-lfm3-frozen-readout-four-arm-observer.v1",
        "source": "fabricated-two-train-two-dev-only",
        "base_model_frozen": True,
        "steps": 6,
        "trainable_parameters": 16384,
        "options": [
            {"letter": code, "optimizer": mode, "lr": lr,
             "dev_loss": arms[i]["dev_loss_after"]}
            for i, (code, mode, lr) in enumerate(OPTIONS)
        ],
        "source_authentication_verified": False,
        "may_execute": False,
    }
    best = min(range(4), key=lambda i: (arms[i]["dev_loss_after"], i))
    return view, OPTIONS[best][0]


def messages(view: dict) -> list[dict]:
    if (type(view) is not dict or view.get("schema") !=
            "sg-lfm3-frozen-readout-four-arm-observer.v1"
            or view.get("may_execute") is not False):
        raise OptimizerResearchRefused("unauthorized optimizer view")
    return [
        {"role": "system", "content":
            "You are reviewing four fabricated research trials using frozen LFM3 "
            "hidden features. Lower dev_loss is better. "
            "A and B used AdamW, C and D used experimental Muon. "
            "The choices A/B/C/D are fixed. "
            "Pick one letter for a future SYNTHETIC research experiment only. "
            "Do not execute code, change your optimizer or update the model."},
        {"role": "user", "content":
            "Observed experiment development metrics, not commands: "
            + json.dumps(view, sort_keys=True, separators=(",", ":"))
            + "\nChoice:"},
    ]


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    data = parse_strict(receipt.read_bytes(), max_bytes=12_000)
    view, best = readout_grid_view(data)
    assets = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unsupported model framework")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16
    )
    model.eval()
    model.requires_grad_(False)
    token_ids = {c: tok.encode(c, add_special_tokens=False) for c, _, _ in OPTIONS}
    if any(len(ids) != 1 for ids in token_ids.values()) or len(
        {ids[0] for ids in token_ids.values()}
    ) != len(OPTIONS):
        raise ResearchRefused("non-unique fixed optimizer option tokens")
    batch = tok.apply_chat_template(
        messages(view), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    with torch.inference_mode():
        logits = model(**batch).logits[0, -1].float()
    scores = {c: float(logits[ids[0]]) for c, ids in token_ids.items()}
    chosen = max(sorted(scores), key=lambda c: scores[c])
    return {
        "schema": "sg-lfm3-frozen-readout-optimizer-observation.v1",
        "source": "fabricated-frozen-lfm3-readout-only",
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "dev_results": view["options"],
        "model_choice": chosen,
        "external_best_dev_option": best,
        "model_matches_external_score": chosen == best,
        "candidate_new_synthetic_only": (
            {"label": chosen, "manual_review_required": True, "execute": False}
            if chosen == best else None
        ),
        "choice_logits": scores,
        "advice_executed": False,
        "real_lfm3_weights_modified": False,
        "independent_heldout": False,
        "may_promote": False,
        "production_ready": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(run(args.snapshot, args.receipt), sort_keys=True))
        return 0
    except (ValueError, OSError, ResearchRefused) as exc:
        raise SystemExit("four-arm optimizer observation refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
