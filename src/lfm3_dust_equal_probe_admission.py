"""Three-seed LFM3 neural-v-random equal-one-probe quality admission.

Exactly one training perturbation and six diagnostic forwards per policy;
neural pays separate frozen-head reconstruction/context/inference overhead.
Every result is diagnostic evidence, not admission for live LoRA updates.
"""
from __future__ import annotations
import statistics
from src.lfm3_dust_equal_probe_review import review

SCHEMA="sensorgraft.lfm3.equal-probe-admission.v1"
SEEDS=(25,26,27)
POLICIES=("random_one_probe","neural_one_probe")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False}


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=3:
        return blocked("three_new_pretrained_model_run_receipts_required")
    source={}
    unique_runs=set()
    for raw in reports:
        result=review(raw,repo_root=repo_root)
        if result.get("status")!="manual_matched_probe_budget_review_only":
            return blocked("unverifiable_model_head_or_one_probe_policy_execution")
        seed=result["seed"]
        if seed in source or raw["run_id"] in unique_runs:
            return blocked("reused_seed_or_execution_run")
        source[seed]=result
        unique_runs.add(raw["run_id"])
    if set(source)!=set(SEEDS):
        return blocked("wrong_seed_set")
    outcomes={}
    for policy in POLICIES:
        items=[source[s]["results"][policy] for s in SEEDS]
        d=[x["diagnostic_loss_delta"] for x in items]
        t=[x["transfer_loss_delta"] for x in items]
        same=[x["same_family_loss_delta"] for x in items]
        elapsed=[x["probe_plus_diagnostic_elapsed_s"] for x in items]
        outcomes[policy]={
            "mean_diagnostic_loss_delta":statistics.mean(d),
            "per_seed_diagnostic_loss_delta":dict(zip(map(str,SEEDS),d,strict=True)),
            "per_seed_transfer_loss_delta":dict(zip(map(str,SEEDS),t,strict=True)),
            "per_seed_same_family_loss_delta":dict(zip(map(str,SEEDS),same,strict=True)),
            "one_step_updates_temporarily_evaluated":sum(
                x["temporary_adapter_step_accepted"] for x in items),
            "policy_probe_plus_diagnostic_elapsed_s":dict(zip(
                map(str,SEEDS),elapsed,strict=True)),
            "training_probe_forwards_per_seed":1,
            "diagnostic_forwards_per_seed":6,
        }
    neural=outcomes["neural_one_probe"]
    random=outcomes["random_one_probe"]
    reasons=[]
    if not all(x<-1e-6 for x in neural["per_seed_diagnostic_loss_delta"].values()):
        reasons.append("not_consistent_independent_diagnostic_improvement_each_seed")
    if any(x>1e-6 for x in neural["per_seed_transfer_loss_delta"].values()):
        reasons.append("unseen_uppercase_task_loss_increased")
    if neural["mean_diagnostic_loss_delta"]>=random["mean_diagnostic_loss_delta"]:
        reasons.append("mean_neural_loss_not_better_than_random")
    if neural["one_step_updates_temporarily_evaluated"]<3:
        reasons.append("insufficient_accepted_neural_updates_for_quality_claim")
    # Additional model forward and source head reconstruction mean the total
    # compute is NOT matched even though candidate probes are exactly matched.
    reasons.append("neural_extra_attention_context_and_head_compute_not_resource_matched")
    return {
        "schema":SCHEMA,"status":"hold_not_qualified",
        "blocking_reasons":reasons,
        "seeds":list(SEEDS),
        "method":"predictive_dust",
        "external_trained_neural_head_parameters":195,
        "model_lora_b_parameters":2048,
        "total_actual_model_completion_loss_forwards":63,
        "total_model_prompt_only_attention_context_forwards":3,
        "total_model_candidate_probes_random":3,
        "total_model_candidate_probes_neural":3,
        "model_candidate_probe_budgets_equal":True,
        "total_walltime_compute_resource_matched":False,
        "no_update_baseline_diagnostic_loss_delta":0.,
        "source_execution_cryptographically_attested":False,
        "underlying_task_families":["first_word","last_word","uppercase"],
        "policy_outcomes":outcomes,
        "model_weights_changed_persistently":False,
        "source_admission_to_live_optimizer":False,
        "model_training_permitted":False,
        "optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
    }
