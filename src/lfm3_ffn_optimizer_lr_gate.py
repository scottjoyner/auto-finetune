"""Read-only numeric gate for 15 actual LFM3 FFN-LoRA AdamW/Muon LR trials.

No holdout exists for this new development-calibrated choice. An observing LFM3
may recommend a fixed NEXT SYNTHETIC trial only; the actual optimizer, source,
model weights and evaluator cannot be changed through this code.
"""
from __future__ import annotations
import math
import statistics
from src.lfm3_ffn_multiseed_guard import admit_three_seed_training
from src.lfm3_real_checkpoint_smoke import ResearchRefused

PIN="135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
REV="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
TARGET="model.layers.23.feed_forward.w2"
SEEDS=(20261028,20261029,20261030)
ARMS=(("adamw",0.001),("adamw",0.002),("adamw",0.004),
      ("muon_match_rms_adamw",0.002),("muon_match_rms_adamw",0.004))
SCHEMA="sg-lfm3-real-ffn-optimizer-lr-calibration-3seed.v1"
REPORT_FIELDS={
    "schema","checkpoint_sha256","model_revision","target_module",
    "pretrained_parameters_frozen","trainable_adapter_parameters",
    "allowed_arms","seeds","steps_per_arm","train_sha256","dev_sha256",
    "only_two_fabricated_train_and_two_dev","holdout_evaluated",
    "source_authenticated","model_advice_used_for_training","trials",
    "base_model_weights_modified","adapter_saved","optimizer_reconfigured_live",
    "may_auto_execute","may_promote_fleet","production_ready",
    "experiment_date","node","offline_bwrap_enforced","independently_attested",
    "independent_task_heldout_evaluated","learning_rate_promoted",
}
ARM_FIELDS={
    "seed","optimizer","learning_rate","steps","target","rank",
    "trainable_adapter_parameters","train_loss_before","dev_loss_before",
    "train_loss_after","dev_loss_after","max_gradient_norm_before_clipping",
    "full_model_gradient_verified","pretrained_backbone_modified",
    "adapter_checkpoint_written",
}
MIN_GAIN=0.05

def gate_calibration(report: object, original: object) -> dict:
    old=admit_three_seed_training(original)
    if old["fixed_objective_best"]!="A":
        raise ResearchRefused("prior AdamW-vs-Muon development comparison differs")
    if type(report) is not dict or set(report)!=REPORT_FIELDS:
        raise ResearchRefused("unknown LFM3 calibration result schema")
    if (report["schema"]!=SCHEMA or report["checkpoint_sha256"]!=PIN
            or report["model_revision"]!=REV or report["target_module"]!=TARGET
            or type(report["pretrained_parameters_frozen"]) is not int
            or report["pretrained_parameters_frozen"]!=1_728_801_536
            or type(report["trainable_adapter_parameters"]) is not int
            or report["trainable_adapter_parameters"]!=38_912
            or report["seeds"]!=list(SEEDS)
            or report["allowed_arms"]!=[{"optimizer":opt,"learning_rate":lr}
                                       for opt,lr in ARMS]
            or type(report["steps_per_arm"]) is not int
            or report["steps_per_arm"]!=3
            or report["train_sha256"]!=original["synthetic_train_sha256"]
            or report["dev_sha256"]!=original["synthetic_dev_sha256"]
            or report["only_two_fabricated_train_and_two_dev"] is not True
            or report["holdout_evaluated"] is not False
            or report["source_authenticated"] is not False
            or report["model_advice_used_for_training"] is not False
            or report["base_model_weights_modified"] is not False
            or report["adapter_saved"] is not False
            or report["optimizer_reconfigured_live"] is not False
            or report["may_auto_execute"] is not False
            or report["may_promote_fleet"] is not False
            or report["production_ready"] is not False
            or report["offline_bwrap_enforced"] is not True
            or report["independently_attested"] is not False
            or report["independent_task_heldout_evaluated"] is not False
            or report["learning_rate_promoted"] is not False
            or report["experiment_date"]!="2026-10-28"
            or report["node"]!="xwing"):
        raise ResearchRefused("wrong pinned source or elevated optimizer authority")
    rows=report["trials"]
    if type(rows) is not list or len(rows)!=15:
        raise ResearchRefused("calibration requires 15 actual paired trials")
    losses=[[None]*5 for _ in SEEDS]
    for i,seed in enumerate(SEEDS):
        previous=None
        for j,(opt,lr) in enumerate(ARMS):
            arm=rows[i*5+j]
            if (type(arm) is not dict or set(arm)!=ARM_FIELDS
                    or type(arm["seed"]) is not int or arm["seed"]!=seed
                    or arm["optimizer"]!=opt or type(arm["learning_rate"]) is not float
                    or arm["learning_rate"]!=lr or type(arm["steps"]) is not int
                    or arm["steps"]!=3 or arm["target"]!=TARGET
                    or type(arm["rank"]) is not int or arm["rank"]!=4
                    or type(arm["trainable_adapter_parameters"]) is not int
                    or arm["trainable_adapter_parameters"]!=38_912
                    or arm["full_model_gradient_verified"] is not True
                    or arm["pretrained_backbone_modified"] is not False
                    or arm["adapter_checkpoint_written"] is not False):
                raise ResearchRefused("noncanonical optimizer seed, lr or gradient evidence")
            for field in ("train_loss_before","dev_loss_before","train_loss_after",
                          "dev_loss_after","max_gradient_norm_before_clipping"):
                n=arm[field]
                if type(n) not in (int,float) or not math.isfinite(n) or n<0:
                    raise ResearchRefused("nonfinite FFN calibration metric")
            start=(arm["train_loss_before"],arm["dev_loss_before"])
            if previous is not None and any(abs(a-b)>0.0001 for a,b in zip(start,previous)):
                raise ResearchRefused("different baseline across the fixed optimizer arms")
            previous=start
            if j in (1,3):
                old_arm=original["arms"][i*2+(0 if j==1 else 1)]
                if (abs(arm["dev_loss_after"]-old_arm["dev_loss_after"])>0.01
                        or abs(arm["train_loss_after"]-old_arm["train_loss_after"])>0.01):
                    raise ResearchRefused("new LR experiment does not reproduce prior LR 0.002")
            losses[i][j]=arm["dev_loss_after"]
    best_per_seed=[min(range(5),key=lambda col:losses[i][col]) for i in range(3)]
    avg=[statistics.mean(losses[i][col] for i in range(3)) for col in range(5)]
    ranked=sorted(range(5),key=lambda i:(avg[i],i))
    winner=ranked[0]
    previous=[losses[i][1] for i in range(3)]
    improvement=[(previous[i]-losses[i][winner])/max(previous[i],1e-12) for i in range(3)]
    review=all(x==winner for x in best_per_seed) and all(x>=MIN_GAIN for x in improvement)
    return {
        "schema":"sg-lfm3-real-ffn-lr-calibration-assessment.v1",
        "seeds":list(SEEDS),
        "options":[{"optimizer":opt,"learning_rate":lr} for opt,lr in ARMS],
        "development_loss_by_seed":losses,
        "mean_development_loss_by_option":avg,
        "best_option_index":winner,
        "best_option_each_seed":best_per_seed,
        "relative_improvement_over_adamw_002_by_seed":improvement,
        "development_only_research_review_eligible":review,
        "heldout_evaluated":False,
        "independent_task_family_covered":False,
        "may_execute_experiment":False,
        "may_adjust_live_optimizer":False,
        "may_save_weights":False,
        "may_promote":False,
        "production_ready":False,
    }
