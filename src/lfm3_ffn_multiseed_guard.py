"""Deterministic three-seed *real LFM3 graph* FFN optimizer research gate.

Never lets an LLM set source permissions, dev labels, seed identity, trainable
parameter groups, LR, optimizer state or evaluation thresholds. Validates a
fixed set of six previously measured FFN-LoRA trials. No training happens here.
"""
from __future__ import annotations
import math
import statistics
from src.lfm3_real_checkpoint_smoke import ResearchRefused

SEEDS=(20261028,20261029,20261030)
MODES=("adamw","muon_match_rms_adamw")
PIN="135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
REV="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
SCHEMA="sg-lfm3-real-last-layer-lora-3seed-3step.v1"
TARGET="model.layers.23.feed_forward.w2"
ARM_FIELDS={
 "seed","optimizer","learning_rate","steps","target","rank",
 "trainable_adapter_parameters","train_loss_before","train_loss_after",
 "dev_loss_before","dev_loss_after","gradient_norm_before_clipping",
 "input_matrix_updated","verified_real_model_gradient",
 "base_weight_update_executed","adapter_checkpoint_written",
}
REPORT_FIELDS={
 "schema","checkpoint_sha256","model_revision","target_module",
 "full_frozen_model_forward_used","gradient_through_real_last_layer",
 "backbone_parameters_frozen","synthetic_train_sha256","synthetic_dev_sha256",
 "two_train_two_dev_fabricated_only","arms","optimizer_model_chosen",
 "train_steps_per_arm","research_seeds","trial_count","per_seed_adamw_wins",
 "eligible_for_manual_review_only","heldout_evaluated",
 "source_custody_independently_verified","base_model_weights_modified",
 "adapter_weights_saved","may_promote_fleet","production_ready",
 "experiment_date","node","offline_bwrap_enforced","independently_attested",
 "independent_heldout_evaluated","source_generation_verified",
}


def admit_three_seed_training(report: object) -> dict:
    if type(report) is not dict or set(report) != REPORT_FIELDS:
        raise ResearchRefused("unknown real LFM3 FFN multi-seed receipt schema")
    if (report["schema"] != SCHEMA
            or report["checkpoint_sha256"] != PIN
            or report["model_revision"] != REV
            or report["target_module"] != TARGET
            or type(report["backbone_parameters_frozen"]) is not int
            or report["backbone_parameters_frozen"] != 1728801536
            or report["full_frozen_model_forward_used"] is not True
            or report["gradient_through_real_last_layer"] is not True
            or report["two_train_two_dev_fabricated_only"] is not True
            or report["optimizer_model_chosen"] is not False
            or type(report["train_steps_per_arm"]) is not int
            or report["train_steps_per_arm"] != 3
            or report["research_seeds"] != list(SEEDS)
            or type(report["trial_count"]) is not int
            or report["trial_count"] != 6
            or report["heldout_evaluated"] is not False
            or report["source_custody_independently_verified"] is not False
            or report["base_model_weights_modified"] is not False
            or report["adapter_weights_saved"] is not False
            or report["may_promote_fleet"] is not False
            or report["production_ready"] is not False
            or report["experiment_date"] != "2026-10-28"
            or report["node"] != "xwing"
            or report["offline_bwrap_enforced"] is not True
            or report["independently_attested"] is not False
            or report["independent_heldout_evaluated"] is not False
            or report["source_generation_verified"] is not False):
        raise ResearchRefused("training/source/promotion contract mismatch")
    for name in ("synthetic_train_sha256","synthetic_dev_sha256"):
        value=report[name]
        if type(value) is not str or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ResearchRefused("noncanonical synthetic data identity")
    if report["synthetic_train_sha256"]==report["synthetic_dev_sha256"]:
        raise ResearchRefused("train/dev split alias")
    arms=report["arms"]
    if type(arms) is not list or len(arms)!=6:
        raise ResearchRefused("fixed six measured trial rows required")
    dev_pairs=[]
    for slot,seed in enumerate(SEEDS):
        pair=[]
        for i,mode in enumerate(MODES):
            arm=arms[2*slot+i]
            if type(arm) is not dict or set(arm) != ARM_FIELDS:
                raise ResearchRefused("real FFN experiment arm fields mismatched")
            if (type(arm["seed"]) is not int or arm["seed"] != seed
                    or arm["optimizer"] != mode
                    or type(arm["learning_rate"]) is not float or arm["learning_rate"]!=0.002
                    or type(arm["steps"]) is not int or arm["steps"] != 3
                    or arm["target"] != TARGET
                    or type(arm["rank"]) is not int or arm["rank"]!=4
                    or type(arm["trainable_adapter_parameters"]) is not int
                    or arm["trainable_adapter_parameters"]!=38912
                    or arm["input_matrix_updated"] is not True
                    or arm["verified_real_model_gradient"] is not True
                    or arm["base_weight_update_executed"] is not False
                    or arm["adapter_checkpoint_written"] is not False):
                raise ResearchRefused("unpaired/forged real FFN training provenance")
            for n in ("train_loss_before","train_loss_after","dev_loss_before","dev_loss_after","gradient_norm_before_clipping"):
                x=arm[n]
                if type(x) not in (float,int) or not math.isfinite(x) or x < 0:
                    raise ResearchRefused("nonfinite/invalid FFN optimization telemetry")
            pair.append(arm)
        if (abs(pair[0]["train_loss_before"]-pair[1]["train_loss_before"])>1e-4
                or abs(pair[0]["dev_loss_before"]-pair[1]["dev_loss_before"])>1e-4):
            raise ResearchRefused("optimizer candidates have different initial losses")
        dev_pairs.append((pair[0]["dev_loss_after"],pair[1]["dev_loss_after"]))
    adamw_better=[a<b for a,b in dev_pairs]
    if report["per_seed_adamw_wins"] != adamw_better:
        raise ResearchRefused("reported seed results do not match measured development losses")
    if report["eligible_for_manual_review_only"] is not all(adamw_better):
        raise ResearchRefused("reported next-trial admission disagrees with independent gate")
    baseline=[arms[i*2]["dev_loss_before"] for i in range(3)]
    if max(baseline)-min(baseline)>1e-4:
        raise ResearchRefused("seeds did not share a fixed frozen pretrained baseline")
    median_adamw=statistics.median(a for a,_ in dev_pairs)
    median_muon=statistics.median(b for _,b in dev_pairs)
    improvement=[(b-a)/max(b,1e-12) for a,b in dev_pairs]
    meaningful=all(x>=0.02 for x in improvement)
    candidate=all(adamw_better) and meaningful
    return {
        "schema":"sg-lfm3-ffn-multiseed-assessment.v1",
        "training_schema":SCHEMA,
        "seed_count":3, "arm_count":6,
        "optimizer_development_median_adamw":median_adamw,
        "optimizer_development_median_muon_rms":median_muon,
        "paired_relative_dev_gain_by_seed":improvement,
        "adamw_lower_dev_each_seed":adamw_better,
        "minimum_two_percent_gain_all_seeds":meaningful,
        "fixed_objective_best":"A" if median_adamw<median_muon else "B",
        "may_propose_next_synthetic_review_only":candidate,
        "auto_execute":False,"independent_heldout_passed":False,
        "can_modify_optimizers":False,"can_update_model":False,
        "may_promote_fleet":False,"production_ready":False,
    }
