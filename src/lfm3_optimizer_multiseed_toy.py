"""Three-seed fixed AdamW/Muon toy experiment with deterministic dev-only gate.

Designed to reject lucky single-seed optimizer gains. No real LFM3 weights,
no heldout labels, no persistence and no autonomous future run scheduling.
"""
from __future__ import annotations
import json
import math
import statistics
from src.lfm3_optimizer_guard import (
    OptimizerResearchRefused, validate_observation, compare_dev_only
)
from src.lfm3_optimizer_grid_toy import GRID
from src.lfm3_optimizer_toy import run_toy_arm

SEEDS = (17, 71, 151)
BASELINE_INDEX = 1


def assess(trials: object) -> dict:
    if type(trials) is not list or len(trials) != len(SEEDS):
        raise OptimizerResearchRefused("fixed multi-seed run count changed")
    by_arm = {i: [] for i in range(len(GRID))}
    seen_train, seen_dev = set(), set()
    per_seed = []
    for slot, item in enumerate(trials):
        if type(item) is not dict or set(item) != {"seed", "arms"}:
            raise OptimizerResearchRefused("multi-seed entry schema mismatch")
        if type(item["seed"]) is not int or item["seed"] != SEEDS[slot]:
            raise OptimizerResearchRefused("unexpected optimizer seed ordering")
        arms = item["arms"]
        if type(arms) is not list or len(arms) != len(GRID):
            raise OptimizerResearchRefused("missing optimizer arm in fixed grid")
        for i, (mode, lr) in enumerate(GRID):
            entry = validate_observation(arms[i])
            if (entry["seed"] != item["seed"] or entry["steps"] != 24
                    or entry["optimizer"] != mode or entry["learning_rate"] != lr
                    or entry["nonfinite_updates"] != 0):
                raise OptimizerResearchRefused("unpaired/unstable optimizer arm")
            reference = arms[BASELINE_INDEX]
            if (entry["train_session_sha256"] != reference["train_session_sha256"]
                    or entry["dev_session_sha256"] != reference["dev_session_sha256"]):
                raise OptimizerResearchRefused("same-seed trial source mismatch")
            by_arm[i].append(entry["dev_loss"])
        ref = arms[BASELINE_INDEX]
        if ref["train_session_sha256"] in seen_train or ref["dev_session_sha256"] in seen_dev:
            raise OptimizerResearchRefused("seed provenance repeated across trials")
        seen_train.add(ref["train_session_sha256"])
        seen_dev.add(ref["dev_session_sha256"])
        per_seed.append({
            "seed": item["seed"],
            "dev_best_index": min(range(6), key=lambda i: (arms[i]["dev_loss"], i)),
            "adamw_0p001_dev_loss": ref["dev_loss"],
            "adamw_0p002_dev_loss": arms[2]["dev_loss"],
            "muon_0p002_dev_loss": arms[5]["dev_loss"],
        })
    median_dev = {str(i): statistics.median(losses)
                  for i, losses in by_arm.items()}
    best_index = min(range(len(GRID)),
                     key=lambda i: (median_dev[str(i)], i))
    is_consistent = all(
        trials[j]["arms"][best_index]["dev_loss"]
        <= trials[j]["arms"][BASELINE_INDEX]["dev_loss"] * 0.98
        for j in range(len(SEEDS))
    )
    # A fixed multi-seed development pass can only nominate a human-reviewed
    # next experiment. Training this exact toy grid isn't genuine LFM3 tuning.
    return {
        "schema": "sg-lfm3-optimizer-multiseed-summary.v1",
        "source": "fabricated-toy-only",
        "seeds": list(SEEDS),
        "trial_count": len(SEEDS) * len(GRID),
        "grid": [{"optimizer": mode, "learning_rate": lr} for mode, lr in GRID],
        "per_seed": per_seed,
        "median_dev_loss_by_index": median_dev,
        "median_best_grid_index": best_index,
        "median_best_optimizer": GRID[best_index][0],
        "median_best_lr": GRID[best_index][1],
        "best_improves_reference_every_seed_by_two_percent": is_consistent,
        "may_submit_for_manual_research_review": is_consistent,
        "source_generations_authenticated": False,
        "independent_heldout_evaluated": False,
        "may_change_running_optimizer": False,
        "may_train_real_lfm3": False,
        "may_promote_fleet": False,
        "model_weights_written": False,
        "production_ready": False,
    }


def run() -> dict:
    trials = [
        {"seed": seed, "arms": [
            run_toy_arm(mode, steps=24, learning_rate=lr, seed=seed)
            for mode, lr in GRID
        ]}
        for seed in SEEDS
    ]
    return {
        "schema": "sg-lfm3-optimizer-multiseed-toy.v1",
        "fabricated_only": True,
        "actual_lfm3_trained": False,
        "trials": trials,
        "assessment": assess(trials),
        "independent_source_custody_proven": False,
        "checkpoint_written": False,
        "production_ready": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
