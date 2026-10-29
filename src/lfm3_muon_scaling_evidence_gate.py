"""Immutable numeric gate for native Muon scaling on real LFM3 FFN LoRA.

Measured original-vs-RMS differences can be BF16 rounding artifacts. Never
promote an optimizer on a near-tie, and never let an observing LLM change this
gate, the LR, source data, adapter, evaluation budget, or trained weights.
"""
from __future__ import annotations
import math
from statistics import median
from src.lfm3_real_checkpoint_smoke import ResearchRefused

SEEDS = (20261028, 20261029, 20261030)
OPTIONS = ("muon_original", "muon_match_rms_adamw")
PIN = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
REV = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
TARGET = "model.layers.23.feed_forward.w2"
MIN_ABSOLUTE_DEV_GAIN = 0.05
MIN_RELATIVE_DEV_GAIN = 0.005
REPORT_FIELDS = {
    "adapter_weights_saved", "arms", "backbone_parameters_frozen",
    "base_model_weights_modified", "checkpoint_sha256",
    "eligible_for_manual_review_only", "full_frozen_model_forward_used",
    "gradient_through_real_last_layer", "heldout_evaluated",
    "may_nominate_optimizer_from_scaling_study", "may_promote_fleet",
    "model_revision", "muon_scaling_options", "muon_scaling_original_vs_rms",
    "optimizer_model_chosen", "per_seed_original_muon_wins",
    "production_ready", "research_seeds", "schema",
    "source_custody_independently_verified", "synthetic_dev_sha256",
    "synthetic_train_sha256", "target_module", "train_steps_per_arm",
    "trial_count", "two_train_two_dev_fabricated_only", "experiment_date",
    "node", "source_independently_attested", "actual_heldout_evaluated",
    "training_authority_granted",
}
ARM_FIELDS = {
    "adapter_checkpoint_written", "base_weight_update_executed",
    "dev_loss_after", "dev_loss_before", "gradient_norm_before_clipping",
    "input_matrix_updated", "learning_rate", "optimizer", "rank", "seed",
    "steps", "target", "train_loss_after", "train_loss_before",
    "trainable_adapter_parameters", "verified_real_model_gradient",
}


def assess_muon_scaling(report: object) -> dict:
    if type(report) is not dict or set(report) != REPORT_FIELDS:
        raise ResearchRefused("unknown native Muon scaling evidence")
    if (report["schema"] != "sg-lfm3-real-last-layer-muon-scaling-3seed.v1"
            or report["checkpoint_sha256"] != PIN
            or report["model_revision"] != REV
            or report["target_module"] != TARGET
            or type(report["backbone_parameters_frozen"]) is not int
            or report["backbone_parameters_frozen"] != 1_728_801_536
            or report["research_seeds"] != list(SEEDS)
            or report["muon_scaling_options"] != list(OPTIONS)
            or report["muon_scaling_original_vs_rms"] is not True
            or type(report["trial_count"]) is not int
            or report["trial_count"] != 6
            or type(report["train_steps_per_arm"]) is not int
            or report["train_steps_per_arm"] != 3
            or report["full_frozen_model_forward_used"] is not True
            or report["gradient_through_real_last_layer"] is not True
            or report["two_train_two_dev_fabricated_only"] is not True
            or report["eligible_for_manual_review_only"] is not False
            or report["may_nominate_optimizer_from_scaling_study"] is not False
            or report["optimizer_model_chosen"] is not False
            or report["base_model_weights_modified"] is not False
            or report["adapter_weights_saved"] is not False
            or report["heldout_evaluated"] is not False
            or report["actual_heldout_evaluated"] is not False
            or report["source_custody_independently_verified"] is not False
            or report["source_independently_attested"] is not False
            or report["training_authority_granted"] is not False
            or report["may_promote_fleet"] is not False
            or report["production_ready"] is not False
            or report["experiment_date"] != "2026-10-28"
            or report["node"] != "xwing"):
        raise ResearchRefused("model, source, trust, or optimizer budget mismatch")
    for key in ("synthetic_train_sha256", "synthetic_dev_sha256"):
        value = report[key]
        if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value):
            raise ResearchRefused("invalid synthetic split identity")
    if report["synthetic_train_sha256"] == report["synthetic_dev_sha256"]:
        raise ResearchRefused("train/dev split alias")
    arms = report["arms"]
    if type(arms) is not list or len(arms) != 6:
        raise ResearchRefused("expected exactly three paired Muon scaling seeds")
    rows = []
    origins = []
    for index, seed in enumerate(SEEDS):
        pair = []
        for offset, mode in enumerate(OPTIONS):
            arm = arms[2 * index + offset]
            if (type(arm) is not dict or set(arm) != ARM_FIELDS
                    or type(arm["seed"]) is not int or arm["seed"] != seed
                    or arm["optimizer"] != mode
                    or type(arm["learning_rate"]) is not float
                    or arm["learning_rate"] != 0.002
                    or type(arm["steps"]) is not int or arm["steps"] != 3
                    or type(arm["rank"]) is not int or arm["rank"] != 4
                    or type(arm["trainable_adapter_parameters"]) is not int
                    or arm["trainable_adapter_parameters"] != 38_912
                    or arm["target"] != TARGET
                    or arm["verified_real_model_gradient"] is not True
                    or arm["input_matrix_updated"] is not True
                    or arm["base_weight_update_executed"] is not False
                    or arm["adapter_checkpoint_written"] is not False):
                raise ResearchRefused("unreviewed or malformed native Muon training arm")
            for name in (
                "dev_loss_before", "dev_loss_after", "train_loss_before",
                "train_loss_after", "gradient_norm_before_clipping"
            ):
                value = arm[name]
                if (type(value) not in (int, float)
                        or not math.isfinite(value) or value < 0):
                    raise ResearchRefused("nonfinite optimizer telemetry")
            pair.append(arm)
        if (abs(pair[0]["dev_loss_before"] - pair[1]["dev_loss_before"]) > 0.0001
                or abs(pair[0]["train_loss_before"] - pair[1]["train_loss_before"]) > 0.0001):
            raise ResearchRefused("Muon's two scaling modes have different initial state")
        origins.append(pair[0]["dev_loss_before"])
        rows.append([pair[0]["dev_loss_after"], pair[1]["dev_loss_after"]])
    if max(origins) - min(origins) > 0.0001:
        raise ResearchRefused("seeds used inconsistent base model output")
    wins = [pair[0] < pair[1] for pair in rows]
    if report["per_seed_original_muon_wins"] != wins:
        raise ResearchRefused("forged reported optimizer ranking")
    effects = [
        abs(original - rms) for original, rms in rows
    ]
    relative = [
        effect / max(min(pair), 1e-12)
        for effect, pair in zip(effects, rows)
    ]
    meaningful = all(
        absolute >= MIN_ABSOLUTE_DEV_GAIN
        and fraction >= MIN_RELATIVE_DEV_GAIN
        for absolute, fraction in zip(effects, relative)
    )
    return {
        "schema": "sg-lfm3-muon-scaling-evidence-gate.v1",
        "seed_count": 3,
        "total_training_trials": 6,
        "median_dev_original": median(row[0] for row in rows),
        "median_dev_rms": median(row[1] for row in rows),
        "absolute_dev_gap_per_seed": effects,
        "relative_dev_gap_per_seed": relative,
        "all_seed_gap_meaningful": meaningful,
        "evidence_warrants_new_research_review": meaningful,
        "tiny_change_conclusion": not meaningful,
        "model_observation_allowed_read_only": True,
        "may_change_optimizer": False,
        "may_rewrite_evaluator": False,
        "may_save_weights": False,
        "may_promote": False,
        "production_ready": False,
    }
