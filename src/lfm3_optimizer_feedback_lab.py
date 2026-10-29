"""Second observational cycle: LFM3 ranks actual synthetic grid dev metrics.

Its token-logit choice is compared to a deterministic argmin on dev loss;
unfounded recommendations are refused. NEVER optimizes LFM3 itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_choice_lab import CHOICES, admit_choice, choice_messages
from src.lfm3_optimizer_guard import (
    OptimizerResearchRefused, observer_view, parse_strict, validate_observation,
    validate_proposal,
)
from src.lfm3_optimizer_grid_toy import GRID
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment


def summarize_grid(report: object) -> tuple[dict, list[dict], str]:
    if type(report) is not dict or (
        report.get("schema") != "sg-lfm3-optimizer-fixed-grid.v1"
        or report.get("source") != "fabricated-toy-only"
        or report.get("experiment_count") != 6
        or report.get("trainable_real_lfm3_parameters") != 0
        or report.get("heldout_evaluated") is not False
        or report.get("cannot_adapt_live_optimizer") is not True
        or report.get("cannot_promote_to_fleet") is not True
        or report.get("model_weights_written") is not False
        or report.get("production_ready") is not False
        or report.get("baseline_index") != 1
        or report.get("fixed_training_steps") != 24
        or report.get("optimizer_runtime") != "native_torch_cpu"
        or report.get("matching_start_weights") is not True
        or report.get("matching_train_and_dev_data") is not True
    ):
        raise OptimizerResearchRefused("not a fixed six-arm synthetic optimization grid")
    trials = report.get("trials")
    if type(trials) is not list or len(trials) != 6:
        raise OptimizerResearchRefused("missing paired trial records")
    for i, (opt, lr) in enumerate(GRID):
        trial = validate_observation(trials[i])
        if (trial["optimizer"] != opt or trial["learning_rate"] != lr
                or trial["steps"] != 24 or trial["nonfinite_updates"] != 0):
            raise OptimizerResearchRefused("optimizer grid changed from fixed protocol")
        if i != 1 and (trial["seed"] != trials[1]["seed"]
                  or trial["train_session_sha256"] != trials[1]["train_session_sha256"]
                  or trial["dev_session_sha256"] != trials[1]["dev_session_sha256"]):
            raise OptimizerResearchRefused("toy grid train/dev session mismatch")
    best_index = min(range(6), key=lambda i: (trials[i]["dev_loss"], i))
    # Inference receives ONLY optimizer/rate/dev numeric summaries, not hashes
    # of source sessions, raw records, heldout targets or model responses.
    summaries = [
        {"option": chr(65+i), "optimizer": t["optimizer"],
         "learning_rate": t["learning_rate"], "dev_loss": round(t["dev_loss"], 8)}
        for i, t in enumerate(trials)
    ]
    return trials[1], summaries, chr(65 + best_index)


def grid_messages(baseline: dict, scores: list[dict]) -> list[dict]:
    view = observer_view(baseline)
    msgs = choice_messages(view)
    msgs[-1]["content"] += (
        "\nSix separately measured, fabricated development losses from fresh "
        "identical model initializations. Lower dev_loss is better. "
        "No heldout values are available. "
        + json.dumps(scores, separators=(",", ":"), sort_keys=True)
        + "\nPick exactly one of A/B/C/D/E/F:"
    )
    return msgs


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    report = parse_strict(receipt.read_bytes(), max_bytes=24_000)
    parent, observations, best = summarize_grid(report)
    assets = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unsupported unpinned Transformer runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16
    )
    model.eval()
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("model class unexpected")
    code_tokens = {code: tok.encode(code, add_special_tokens=False) for code in CHOICES}
    if (any(len(xs) != 1 for xs in code_tokens.values())
            or len({xs[0] for xs in code_tokens.values()}) != len(code_tokens)):
        raise ResearchRefused("optimizer choices not single-token")
    batch = tok.apply_chat_template(
        grid_messages(parent, observations), tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**batch).logits[0, -1].float()
    scores = {code: float(logits[tokens[0]]) for code, tokens in code_tokens.items()}
    choice = max(sorted(scores), key=lambda k: scores[k])
    match = choice == best
    if match:
        candidate = admit_choice(choice, parent)
        candidate["cycle_index"] = 2
        validate_proposal(candidate, parent)
    else:
        candidate = None
    return {
        "schema": "sg-lfm3-optimizer-second-observation.v1",
        "source": "fabricated-toy-grid-only",
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "parent_observation_sha256": observer_view(parent)["observation_sha256"],
        "model_choice": choice,
        "deterministic_dev_loss_best_option": best,
        "model_choice_matches_measured_dev_best": match,
        "choice_logit_scores": scores,
        "admitted_next_synthetic_trial_only": candidate,
        "model_generated_command_executed": False,
        "optimizer_state_changed": False,
        "real_lfm3_weights_trained": False,
        "heldout_evaluated": False,
        "production_ready": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(run(a.snapshot, a.grid), sort_keys=True))
        return 0
    except (ValueError, OSError, ResearchRefused) as exc:
        raise SystemExit("second optimizer observer refused: " + str(exc)) from None


if __name__ == "__main__":
    main()
