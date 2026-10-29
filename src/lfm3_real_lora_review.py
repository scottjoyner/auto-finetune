"""Pure-Python, fail-closed review of real LFM3 / synthetic-token LoRA probes."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
BASE = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"


def blocked(reason):
    return {
        "schema": "sensorgraft.lfm3.real-lora-review.v1",
        "status": "blocked",
        "reason": reason,
        "weight_promotion_permitted": False,
        "proposal": None,
    }


def review(report):
    required = {
        "schema": "sensorgraft.lfm3.real-lora-synthetic-probe.v1",
        "status": "research_synthetic_tokens_real_lfm3",
        "model": "LiquidAI/LFM3-1.7B",
        "base_revision": REVISION,
        "base_sha256": BASE,
        "real_pretrained_model": True,
        "synthetic_only": True,
        "real_private_traces_used": False,
        "heldout_used_for_optimizer_selection": True,
        "optimizer_parameter_policy": "experimental_muon_on_lora_factors",
        "production_ready": False,
        "weight_promotion_permitted": False,
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in required.items()):
        return blocked("unverified_model_or_promotion_claim")
    seeds, steps, arms = report.get("seeds"), report.get("steps"), report.get("arms")
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(steps) is not int
        or not 1 <= steps <= 30
        or not isinstance(arms, list)
        or len(arms) != 2 * len(seeds)
    ):
        return blocked("invalid_seed_step_or_arm_inventory")
    groups, all_runs = {}, set()
    for arm in arms:
        if not isinstance(arm, dict):
            return blocked("invalid_arm")
        seed, name, lr, run_id = (
            arm.get("seed"),
            arm.get("optimizer"),
            arm.get("learning_rate"),
            arm.get("run_id"),
        )
        if (
            type(seed) is not int
            or seed not in seeds
            or name not in ("adamw", "muon")
            or type(lr) not in (float, int)
            or not math.isfinite(lr)
            or not (1e-6 <= lr <= 2e-4 if name == "adamw" else 1e-5 <= lr <= 2e-3)
            or not isinstance(run_id, str)
            or not HEX32.fullmatch(run_id)
            or run_id in all_runs
            or arm.get("steps") != steps
            or arm.get("weight_promotion_permitted") is not False
            or arm.get("frozen_backbone_sample_unchanged") is not True
            or arm.get("trainable_parameters") != 146432
            or arm.get("base_sha256") != BASE
        ):
            return blocked("invalid_arm_safety_or_identity")
        all_runs.add(run_id)
        for k in (
            "initial_adapter_sha256",
            "final_adapter_sha256",
            "train_sha256",
            "heldout_sha256",
        ):
            x = arm.get(k)
            if not isinstance(x, str) or not HEX64.fullmatch(x):
                return blocked("invalid_adapter_or_data_digest")
        if arm["initial_adapter_sha256"] == arm["final_adapter_sha256"]:
            return blocked("no_trainable_adapter_update")
        for k in ("heldout_loss_before", "heldout_loss_after", "relative_heldout_loss_change"):
            x = arm.get(k)
            if type(x) not in (float, int) or not math.isfinite(x):
                return blocked("invalid_holdout_value")
        if (
            arm["heldout_loss_before"] <= 0
            or arm["heldout_loss_after"] <= 0
            or abs(
                arm["heldout_loss_after"] / arm["heldout_loss_before"]
                - 1
                - arm["relative_heldout_loss_change"]
            )
            > 1e-9
        ):
            return blocked("inconsistent_holdout_change")
        events = arm.get("events")
        if not isinstance(events, list) or len(events) != steps:
            return blocked("incomplete_step_telemetry")
        for i, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {
                "schema",
                "run_id",
                "seed",
                "step",
                "base_sha256",
                "dataset_sha256",
                "optimizer",
                "learning_rate",
                "train_loss",
                "grad_norm",
                "update_norm",
                "elapsed_s",
            }:
                return blocked("invalid_step_schema")
            if (
                event.get("schema") != "sensorgraft.lfm3.optimizer-step.v1"
                or event.get("run_id") != run_id
                or event.get("seed") != seed
                or type(event.get("step")) is not int
                or event["step"] != i
                or event.get("base_sha256") != BASE
                or event.get("dataset_sha256") != arm["train_sha256"]
                or event.get("optimizer") != name
                or event.get("learning_rate") != lr
            ):
                return blocked("step_provenance_mismatch")
            for key in ("train_loss", "grad_norm", "update_norm", "elapsed_s"):
                value = event.get(key)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or (value <= 0 if key in ("elapsed_s", "train_loss") else value < 0)
                ):
                    return blocked("invalid_step_metric")
        groups.setdefault(seed, {})
        if name in groups[seed]:
            return blocked("duplicate_optimizer_arm")
        groups[seed][name] = arm
    if any(set(groups.get(s, {})) != {"adamw", "muon"} for s in seeds):
        return blocked("missing_optimizer_arm")
    paired = []
    for seed in sorted(seeds):
        a, m = groups[seed]["adamw"], groups[seed]["muon"]
        for field in (
            "initial_adapter_sha256",
            "train_sha256",
            "heldout_sha256",
            "heldout_loss_before",
            "steps",
        ):
            if a[field] != m[field]:
                return blocked("unpaired_starting_weights_or_dataset")
        paired.append(
            {
                "seed": seed,
                "adamw_heldout": a["heldout_loss_after"],
                "muon_heldout": m["heldout_loss_after"],
                "muon_minus_adamw": m["heldout_loss_after"] - a["heldout_loss_after"],
            }
        )
    return {
        "schema": "sensorgraft.lfm3.real-lora-review.v1",
        "status": "review_required",
        "model": "LiquidAI/LFM3-1.7B",
        "synthetic_only": True,
        "seed_count": len(seeds),
        "steps_per_arm": steps,
        "mean_adamw_heldout_loss": statistics.mean(p["adamw_heldout"] for p in paired),
        "mean_muon_heldout_loss": statistics.mean(p["muon_heldout"] for p in paired),
        "per_seed": paired,
        "proposal": {
            "action": "collect_new_independent_real_data_before_tuning",
            "optimizer_switch_permitted": False,
            "muon_on_lora_factors_validated": False,
        },
        "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        if args.report.stat().st_size > 300_000:
            result = blocked("oversized_input")
        else:
            result = review(json.loads(args.report.read_text(encoding="utf-8")))
    except (OSError, ValueError, UnicodeError):
        result = blocked("unreadable_report")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()