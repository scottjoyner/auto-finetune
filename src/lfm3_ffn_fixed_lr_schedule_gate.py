"""Independently review 12 real pinned-LFM3 optimizer schedule trials.

Validates exact precommitted 3 seeds x 4 schedules, three real gradient
measurements, unchanged pretrained model, and parity with earlier fixed-LR
calibration. No holdout, self-directed training, or optimization authority.
"""
from __future__ import annotations

import math
import statistics

from src.lfm3_ffn_optimizer_lr_gate import gate_calibration
from src.lfm3_ffn_fixed_lr_schedule import SCHEDULES, MODEL_PIN
from src.lfm3_last_layer_lora_lab import TARGET, RANK, RESEARCH_SEEDS
from src.lfm3_real_checkpoint_smoke import ResearchRefused

TOP_KEYS = {
    "schema", "checkpoint_sha256", "model_revision", "frozen_backbone_parameters",
    "target", "rank", "trainable_parameters", "seeds", "allowed_schedules",
    "real_model_training_trials", "frozen_dev_loss", "train_sha256", "dev_sha256",
    "trials", "heldout_evaluated", "optimizer_advice_used",
    "learning_rate_schedule_precommitted", "model_weights_saved",
    "new_training_authorized", "automatic_optimizer_switching_authorized",
    "production_ready", "experiment_date", "node", "source_commit",
    "offline_bwrap_enforced", "authentic_producer_witness",
    "independent_task_holdout_evaluated", "model_recommendation_executed",
}
TRIAL_KEYS = {
    "seed", "optimizer", "precommitted_learning_rates", "trainable_parameters",
    "initial_dev_loss", "observations", "final_dev_loss",
    "backbone_modified", "adapter_saved",
}
OBS_KEYS = {
    "step", "learning_rate", "dev_loss_after",
    "training_objective_before_step", "gradient_norm_before_clipping",
    "optimizer_changed_by_model",
}
REV="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
SOURCE="a84d54d668624c373ba7b746120bf9451008396a"
SCHEMA="sg-lfm3-real-final-ffn-native-lr-schedule-v1"


def _finite(value, *, positive=False):
    return (type(value) in (float,int) and math.isfinite(value)
            and (value>0 if positive else value>=0))


def review_fixed_lr_schedules(report: object, calibration: object,
                              original_dev: object) -> dict:
    prior=gate_calibration(calibration,original_dev)
    if (prior["best_option_index"] != 2
            or prior["heldout_evaluated"] is not False
            or prior["may_adjust_live_optimizer"] is not False):
        raise ResearchRefused("previous independent development source not admitted")
    if type(report) is not dict or set(report)!=TOP_KEYS:
        raise ResearchRefused("unknown optimizer schedule evidence")
    if (report["schema"]!=SCHEMA or report["checkpoint_sha256"]!=MODEL_PIN
            or report["model_revision"]!=REV or report["target"]!=TARGET
            or type(report["rank"]) is not int or report["rank"]!=RANK
            or type(report["frozen_backbone_parameters"]) is not int
            or report["frozen_backbone_parameters"]!=1_728_801_536
            or type(report["trainable_parameters"]) is not int
            or report["trainable_parameters"]!=38_912
            or report["seeds"]!=list(RESEARCH_SEEDS)
            or report["allowed_schedules"]!=[
                {"optimizer":kind,"learning_rates":list(s)}
                for kind,s in SCHEDULES]
            or type(report["real_model_training_trials"]) is not int
            or report["real_model_training_trials"]!=12
            or report["train_sha256"]!=calibration["train_sha256"]
            or report["dev_sha256"]!=calibration["dev_sha256"]
            or not _finite(report["frozen_dev_loss"],positive=True)
            or report["heldout_evaluated"] is not False
            or report["optimizer_advice_used"] is not False
            or report["learning_rate_schedule_precommitted"] is not True
            or report["model_weights_saved"] is not False
            or report["new_training_authorized"] is not False
            or report["automatic_optimizer_switching_authorized"] is not False
            or report["production_ready"] is not False
            or report["experiment_date"]!="2026-10-28"
            or report["node"]!="xwing"
            or report["source_commit"]!=SOURCE
            or report["offline_bwrap_enforced"] is not True
            or report["authentic_producer_witness"] is not False
            or report["independent_task_holdout_evaluated"] is not False
            or report["model_recommendation_executed"] is not False):
        raise ResearchRefused("unverified schedule identity or elevated authority")
    trials=report["trials"]
    if type(trials) is not list or len(trials)!=12:
        raise ResearchRefused("twelve independent real training trials required")
    comparisons={}
    for idx,(kind,schedule) in enumerate(SCHEDULES):
        comparisons[f"{kind}:{'/'.join(str(x) for x in schedule)}"]=[]
        for seed_idx,seed in enumerate(RESEARCH_SEEDS):
            trial=trials[seed_idx*4+idx]
            if (type(trial) is not dict or set(trial)!=TRIAL_KEYS
                    or type(trial["seed"]) is not int or trial["seed"]!=seed
                    or trial["optimizer"]!=kind
                    or trial["precommitted_learning_rates"]!=list(schedule)
                    or type(trial["trainable_parameters"]) is not int
                    or trial["trainable_parameters"]!=38_912
                    or trial["backbone_modified"] is not False
                    or trial["adapter_saved"] is not False
                    or not _finite(trial["initial_dev_loss"],positive=True)
                    or not _finite(trial["final_dev_loss"],positive=True)
                    or abs(trial["initial_dev_loss"]-report["frozen_dev_loss"])>0.0001):
                raise ResearchRefused("wrong optimizer schedule, seed or pretrained baseline")
            obs=trial["observations"]
            if type(obs) is not list or len(obs)!=3:
                raise ResearchRefused("incomplete native optimizer gradient telemetry")
            for step,rate in enumerate(schedule):
                item=obs[step]
                if (type(item) is not dict or set(item)!=OBS_KEYS
                        or type(item["step"]) is not int or item["step"]!=step+1
                        or type(item["learning_rate"]) is not float
                        or item["learning_rate"]!=rate
                        or not _finite(item["dev_loss_after"],positive=True)
                        or not _finite(item["training_objective_before_step"])
                        or not _finite(item["gradient_norm_before_clipping"],positive=True)
                        or item["optimizer_changed_by_model"] is not False):
                    raise ResearchRefused("invalid or invented optimizer step telemetry")
            if trial["final_dev_loss"]!=obs[-1]["dev_loss_after"]:
                raise ResearchRefused("final optimizer development loss mismatch")
            # Constant LR 0.004 arms must independently reproduce the earlier
            # fixed 5-arm real-model run; a warm-switch run has no such oracle.
            if idx in (0,2):
                cal_arm=calibration["trials"][seed_idx*5+(2 if idx==0 else 4)]
                if (cal_arm["optimizer"]!=kind
                        or cal_arm["learning_rate"]!=0.004
                        or abs(trial["final_dev_loss"]-cal_arm["dev_loss_after"])>0.01):
                    raise ResearchRefused("constant optimizer diverged from prior real calibration")
            name=f"{kind}:{'/'.join(str(x) for x in schedule)}"
            comparisons[name].append(trial["final_dev_loss"])
    outputs=[]
    for i,kind in ((0,"adamw"),(2,"muon_match_rms_adamw")):
        fixed=[trials[4*seed+i]["final_dev_loss"] for seed in range(3)]
        switched=[trials[4*seed+i+1]["final_dev_loss"] for seed in range(3)]
        results={
            "optimizer":kind,
            "fixed_004_development_losses":fixed,
            "step1_002_then_004_development_losses":switched,
            "fixed_004_better_each_seed":[a<b for a,b in zip(fixed,switched)],
            "switched_minus_fixed_by_seed":[b-a for a,b in zip(fixed,switched)],
            "fixed_mean_dev_loss":statistics.mean(fixed),
            "switched_mean_dev_loss":statistics.mean(switched),
            "switch_improves_all_seeds":all(b<a for a,b in zip(fixed,switched)),
        }
        outputs.append(results)
    # No dependency on a self-observer or on automatic model promotion.
    return {
        "schema":"sg-lfm3-real-ffn-native-lr-schedule-review.v1",
        "checkpoint_sha256":MODEL_PIN,
        "original_backbone_frozen":True,
        "comparisons":outputs,
        "all_fixed_lrs_better_than_switches":
            all(all(x["fixed_004_better_each_seed"]) for x in outputs),
        "learning_rate_changed_during_bounded_training":True,
        "model_selected_lr_change":False,
        "independent_task_generalization_proven":False,
        "authentic_producer_witness":False,
        "heldout_evaluated":False,
        "may_execute_optimizer_proposal":False,
        "may_change_fleet_optimizer":False,
        "may_save_model_weights":False,
        "may_promote":False,
        "production_ready":False,
    }
