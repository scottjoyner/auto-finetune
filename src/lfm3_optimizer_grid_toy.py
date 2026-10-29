"""Finite six-arm synthetic AdamW/Muon learning-rate sweep; NEVER real LFM3.

Every trial initializes fresh toy weights and optimizer state. The sweep is
a fixed research plan, not a repeating model-driven optimizer controller.
Only development metrics are compared. No checkpoint or raw data is written.
"""
from __future__ import annotations
import json
from src.lfm3_optimizer_guard import compare_dev_only, OptimizerResearchRefused
from src.lfm3_optimizer_toy import run_toy_arm

GRID = (
    ("adamw", 0.0005),
    ("adamw", 0.001),
    ("adamw", 0.002),
    ("muon_hybrid", 0.0005),
    ("muon_hybrid", 0.001),
    ("muon_hybrid", 0.002),
)


def run_sweep() -> dict:
    if len(GRID) != 6 or len(set(GRID)) != 6:
        raise OptimizerResearchRefused("optimizer experiment grid changed unexpectedly")
    trials = [run_toy_arm(optimizer, learning_rate=lr) for optimizer, lr in GRID]
    baseline = trials[1]
    decisions = [compare_dev_only(baseline, trial) for trial in trials]
    # Sorting is descriptive; even the best development result cannot authorize
    # a trained adapter or use of heldout sessions.
    order = sorted(range(len(trials)),
                   key=lambda i: (trials[i]["dev_loss"], trials[i]["optimizer"],
                                  trials[i]["learning_rate"]))
    return {
        "schema": "sg-lfm3-optimizer-fixed-grid.v1",
        "source": "fabricated-toy-only",
        "optimizer_runtime": "native_torch_cpu",
        "trainable_real_lfm3_parameters": 0,
        "experiment_count": 6,
        "matching_start_weights": True,
        "matching_train_and_dev_data": True,
        "fixed_training_steps": 24,
        "baseline_index": 1,
        "trials": trials,
        "comparison_to_adamw_lr_0p001": decisions,
        "dev_order_best_to_worst": order,
        "heldout_evaluated": False,
        "cannot_adapt_live_optimizer": True,
        "cannot_promote_to_fleet": True,
        "model_weights_written": False,
        "production_ready": False,
    }


def main():
    print(json.dumps(run_sweep(), sort_keys=True))


if __name__ == "__main__":
    main()
