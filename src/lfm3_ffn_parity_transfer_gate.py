"""Independent audit of frozen LFM3 AdamW LR 0.002 vs 0.004 parity transfer.

Source/parity task family are deliberately synthetic, and this once-viewed
holdout is SPENT for all subsequent optimizer selection. This evaluator is
read-only and never grants a model, optimizer or trial execution capability.
"""
from __future__ import annotations
import hashlib
import math
from src.lfm3_ffn_optimizer_lr_gate import gate_calibration, PIN, REV, TARGET, SEEDS
from src.lfm3_ffn_adamw_parity_transfer import PARITY
from src.lfm3_real_checkpoint_smoke import ResearchRefused

FIELDS={
 "schema","checkpoint_sha256","model_revision","target_module",
 "fixed_calibration_winner_index","seed_count","arm_count",
 "training_example_count","dev_example_count","parity_task_holdout_example_count",
 "synthetic_train_sha256","synthetic_dev_sha256","fictional_parity_holdout_sha256",
 "original_frozen_parity_holdout_loss","trials","only_fictional_data",
 "different_prompt_task_family","independently_curated_benchmark",
 "heldout_used_for_training_or_lr_selection","extra_training_after_holdout",
 "pretrained_weights_modified","model_observer_did_not_control_optimizer",
 "source_authenticated","may_promote","production_ready",
 "experiment_date","node","offline_bwrap_enforced",
 "producer_independently_attested",
 "parity_holdout_spent_for_optimizer_selection",
 "auto_optimizer_update_authorized",
}
ARM_FIELDS={
 "seed","optimizer","learning_rate","three_training_steps",
 "dev_loss_reproduced","precommitted_dev_loss","parity_holdout_loss",
 "weight_update_after_holdout","adapter_saved",
}

def assess_parity_transfer(report: object, calibration: object, original: object) -> dict:
    numeric=gate_calibration(calibration,original)
    if (numeric["best_option_index"]!=2
            or not numeric["development_only_research_review_eligible"]
            or numeric["heldout_evaluated"] is not False):
        raise ResearchRefused("original LR calibration did not preselect option C")
    if type(report) is not dict or set(report)!=FIELDS:
        raise ResearchRefused("unknown one-shot parity evidence schema")
    if (report["schema"]!="sg-lfm3-real-ffn-adamw-parity-transfer-once.v1"
            or report["checkpoint_sha256"]!=PIN or report["model_revision"]!=REV
            or report["target_module"]!=TARGET
            or type(report["fixed_calibration_winner_index"]) is not int
            or report["fixed_calibration_winner_index"]!=2
            or type(report["seed_count"]) is not int or report["seed_count"]!=3
            or type(report["arm_count"]) is not int or report["arm_count"]!=6
            or report["training_example_count"]!=2
            or report["dev_example_count"]!=2
            or report["parity_task_holdout_example_count"]!=4
            or report["synthetic_train_sha256"]!=calibration["train_sha256"]
            or report["synthetic_dev_sha256"]!=calibration["dev_sha256"]
            or report["fictional_parity_holdout_sha256"]!=hashlib.sha256(
                repr(PARITY).encode()).hexdigest()
            or report["only_fictional_data"] is not True
            or report["different_prompt_task_family"] is not True
            or report["independently_curated_benchmark"] is not False
            or report["heldout_used_for_training_or_lr_selection"] is not False
            or report["extra_training_after_holdout"] is not False
            or report["pretrained_weights_modified"] is not False
            or report["model_observer_did_not_control_optimizer"] is not True
            or report["source_authenticated"] is not False
            or report["may_promote"] is not False
            or report["production_ready"] is not False
            or report["experiment_date"]!="2026-10-28"
            or report["node"]!="xwing"
            or report["offline_bwrap_enforced"] is not True
            or report["producer_independently_attested"] is not False
            or report["parity_holdout_spent_for_optimizer_selection"] is not True
            or report["auto_optimizer_update_authorized"] is not False):
        raise ResearchRefused("parity task-family identity or training/promotion claims changed")
    base=report["original_frozen_parity_holdout_loss"]
    if type(base) not in (int,float) or not math.isfinite(base) or base<=0:
        raise ResearchRefused("invalid frozen pretrained parity control")
    arms=report["trials"]
    if type(arms) is not list or len(arms)!=6:
        raise ResearchRefused("expected six three-seed paired AdamW parity outcomes")
    pairs=[]
    for i,seed in enumerate(SEEDS):
        results=[]
        for j,lr in enumerate((0.002,0.004)):
            arm=arms[2*i+j]
            if (type(arm) is not dict or set(arm)!=ARM_FIELDS
                    or type(arm["seed"]) is not int or arm["seed"]!=seed
                    or arm["optimizer"]!="adamw"
                    or type(arm["learning_rate"]) is not float
                    or arm["learning_rate"]!=lr
                    or type(arm["three_training_steps"]) is not int
                    or arm["three_training_steps"]!=3
                    or arm["weight_update_after_holdout"] is not False
                    or arm["adapter_saved"] is not False):
                raise ResearchRefused("forged parity task arm, seed or heldout updates")
            for key in ("dev_loss_reproduced","precommitted_dev_loss","parity_holdout_loss"):
                x=arm[key]
                if type(x) not in (int,float) or not math.isfinite(x) or x<0:
                    raise ResearchRefused("nonfinite one-shot parity task loss")
            expected=calibration["trials"][i*5+(1 if j==0 else 2)]["dev_loss_after"]
            if abs(arm["dev_loss_reproduced"]-expected)>0.01 or abs(
                    arm["precommitted_dev_loss"]-expected)>1e-9:
                raise ResearchRefused("one-shot holdout experiment failed exact dev continuity")
            results.append(arm["parity_holdout_loss"])
        pairs.append(results)
    wins=[pair[1]<pair[0] for pair in pairs]
    gain=[(pair[0]-pair[1])/max(pair[0],1e-12) for pair in pairs]
    gain_vs_frozen=[(base-pair[1])/base for pair in pairs]
    return {
        "schema":"sg-lfm3-adamw-parity-family-audit.v1",
        "seeds":list(SEEDS),
        "frozen_pretrained_parity_cross_entropy":base,
        "lr_002_vs_004_losses_per_seed":pairs,
        "lr_004_better_each_seed":wins,
        "lr_004_relative_gain_vs_002":gain,
        "lr_004_relative_gain_vs_frozen":gain_vs_frozen,
        "different_synthetic_prompt_family":True,
        "holdout_permanently_spent_for_lr_selection":True,
        "independent_curated_benchmark":False,
        "model_observer_was_incorrect":True,
        "model_observer_can_override_numeric_gate":False,
        "may_auto_tune":False,
        "may_train":False,
        "may_promote":False,
        "production_ready":False,
    }
