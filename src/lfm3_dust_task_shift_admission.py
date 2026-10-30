"""Cross-seed admission of trained LFM3 task-shift safeguard, shadow-only.

True-positive novelty detection is assessed on newly generated SORT prompts,
while false-positive checks are historical training leave-one-out similarities
used to define the threshold. They are not independent negative controls.
No model-update permission can be granted through this research admission.
"""
from __future__ import annotations

from src.lfm3_dust_task_shift_review import review

SCHEMA="sensorgraft.lfm3.dust-task-shift-admission.v1"
SEEDS=(13,14,15)


def blocked(reason):
    return {
        "schema":SCHEMA,"status":"blocked","reason":reason,
        "optimizer_switch_permitted":False,"model_update_permitted":False,
        "weight_promotion_permitted":False,
    }


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=3:
        return blocked("exactly_three_new_task_cohorts_required")
    seen={}
    ids=set()
    for report in reports:
        evidence=review(report,repo_root=repo_root)
        if evidence.get("status")!="manual_novelty_shadow_review_only":
            return blocked("unverified_real_pretrained_context")
        seed=evidence["new_task_seed"]
        if seed in seen or report["run_id"] in ids:
            return blocked("reused_task_cohort_or_execution_run")
        ids.add(report["run_id"])
        seen[seed]=evidence
    if set(seen)!=set(SEEDS):
        return blocked("incorrect_new_family_seed_inventory")
    detected=sum(s["new_family_ood_episode_count"] for s in seen.values())
    abstained=sum(s["candidate_abstentions"] for s in seen.values())
    total=18
    return {
        "schema":SCHEMA,"status":"shadow_guard_validated_not_optimizer_qualified",
        "method":"predictive_dust",
        "new_task_family":"sort_two_words",
        "new_task_seed_inventory":list(SEEDS),
        "prompt_only_real_pretrained_model_forwards":18,
        "new_task_prompt_count":total,
        "novel_task_contexts_flagged":detected,
        "novel_task_contexts_not_flagged":total-detected,
        "novelty_detection_fraction":detected/total,
        "train_leave_one_out_examples_used_to_set_threshold":36,
        "candidate_predictions_abstained":abstained,
        "candidate_predictions_total":144,
        "head_training_only_gain_skill_per_seed":{
            str(seed):report["frozen_train_only_guard"]["head_training_gain_skill_pass"]
            for seed,report in
                ((s,next(r for r in reports if r["new_task_seed"]==s))
                 for s in SEEDS)
        },
        "measured_new_family_candidate_gains":0,
        "saved_new_family_perturbation_forwards":None,
        "measured_new_family_predictive_mse":None,
        "actual_model_weight_changes":0,
        "source_runtime_independently_attested":False,
        "novelty_classifier_proves_generalization":False,
        "trained_head_proves_gain_prediction_skill":False,
        "optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }
