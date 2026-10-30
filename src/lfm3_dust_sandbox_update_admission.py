"""Fail-closed selection rule for one-step Dust head-guided quality evidence.

The NEW study tests one ephemeral LoRA-B step on diagnostic prompts, which
did not drive candidate selection, rather than just checking gain forecasts.
No report authorizes production use or independently authenticates the local
LFM3 run. Training acceptance on a single prompt is not sufficient: a
consistent diagnostic improvement is required on all preregistered seeds
and across both task families even to nominate an independent study.
"""
from __future__ import annotations

import statistics

from src.lfm3_dust_sandbox_update_review import review

SCHEMA="sensorgraft.lfm3.dust-one-step-quality-admission.v1"
SEEDS=(16,17,18)
POLICIES=("random_one_probe","neural_one_probe","empirical_best_eight_probes")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,"model_training_permitted":False,
            "weight_promotion_permitted":False}


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=len(SEEDS):
        return blocked("exact_three_new_one_step_evidence_sources_required")
    seen={}
    runs=set()
    for report in reports:
        r=review(report,repo_root=repo_root)
        if r.get("status")!="manual_one_step_counterfactual_review_only":
            return blocked("invalid_original_model_or_neural_head_trajectory")
        seed=r["seed"]
        if seed in seen or report["run_id"] in runs:
            return blocked("duplicate_seed_or_model_trial")
        runs.add(report["run_id"])
        seen[seed]=r
    if set(seen)!=set(SEEDS):
        return blocked("incorrect_single_step_seed_inventory")
    comparison={}
    for name in POLICIES:
        deltas=[seen[s]["results"][name]["diagnostic_loss_delta"] for s in SEEDS]
        same=[seen[s]["results"][name]["same_family_loss_delta"] for s in SEEDS]
        transfer=[seen[s]["results"][name]["transfer_family_loss_delta"] for s in SEEDS]
        measured=[seen[s]["results"][name]["isolated_adapter_step_evaluated"]
                  for s in SEEDS]
        comparison[name]={
            "mean_diagnostic_loss_delta":statistics.mean(deltas),
            "per_seed_loss_deltas":{str(s):deltas[i] for i,s in enumerate(SEEDS)},
            "per_seed_same_family_deltas":{str(s):same[i] for i,s in enumerate(SEEDS)},
            "per_seed_transfer_deltas":{str(s):transfer[i] for i,s in enumerate(SEEDS)},
            "positive_train_gain_temporary_steps":sum(measured),
            "all_seed_mean_diagnostic_improvement":all(d < -1e-6 for d in deltas),
            "all_seed_same_family_improvement":all(d < -1e-6 for d in same),
            "all_seed_transfer_improvement":all(d < -1e-6 for d in transfer),
            "counterfactual_policy_only":True,
        }
    neural=comparison["neural_one_probe"]
    reasons=[]
    if not neural["all_seed_mean_diagnostic_improvement"]:
        reasons.append("neural_one_step_diagnostic_quality_not_consistently_improved")
    if not neural["all_seed_same_family_improvement"]:
        reasons.append("neural_one_step_same_family_quality_not_consistently_improved")
    if not neural["all_seed_transfer_improvement"]:
        reasons.append("neural_one_step_transfer_quality_not_consistently_improved")
    if neural["mean_diagnostic_loss_delta"]>=comparison["random_one_probe"]["mean_diagnostic_loss_delta"]:
        reasons.append("neural_head_not_better_than_random_one_probe")
    return {
        "schema":SCHEMA,
        "status":"hold_not_qualified" if reasons else "separate_confirmation_study_only",
        "blocking_reasons":reasons,
        "seeds":list(SEEDS),
        "source_training_head_seeds":[10,11,12],
        "models_loaded_in_separate_offline_runs":3,
        "one_step_training_prompts":3,
        "diagnostic_prompts_scored_per_trial":6,
        "diagnostic_task_families":["first_word","last_word","uppercase"],
        "total_actual_model_loss_forwards":sum(
            seen[s]["actual_model_loss_forwards"] for s in SEEDS),
        "total_actual_model_context_forwards":3,
        "total_signed_model_candidate_perturbation_forwards":24,
        "total_persistent_model_adapter_updates":0,
        "model_source_runtime_independently_authenticated":False,
        "equal_policy_compute_budget":False,
        "counterfactual_update_evaluation_only":True,
        "admission_to_live_optimizer_from_this_report":False,
        "policy_comparison":comparison,
        "optimizer_switch_permitted":False,
        "model_training_permitted":False,
        "weight_promotion_permitted":False,
    }
