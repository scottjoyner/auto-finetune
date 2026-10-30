"""Admission of *causal*, one-gain-at-a-time LFM3 Q/K head refinement.

The external 195-weight head is genuinely trainable using previous
observed one-probe training outcomes without model backprop. BUT updated
weights must produce independently useful model-choice decisions versus
the unchanged head under the same per-policy candidate-probe budgets.
This tiny synthetic pilot cannot authorize live autonomous fine-tuning.
"""
from __future__ import annotations
import statistics

from src.lfm3_dust_online_one_feedback_review import review

SCHEMA="sensorgraft.lfm3.online-one-feedback-admission.v1"
SEEDS=(28,29,30)
POLICIES=("random_one_probe","frozen_head_one_probe","adaptive_head_one_probe")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,"weight_promotion_permitted":False}


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=3:
        return blocked("three_new_preregistered_lfm3_cohorts_required")
    results={}
    ids=set()
    for raw in reports:
        r=review(raw,repo_root=repo_root)
        if r.get("status")!="manual_causal_one_feedback_head_review_only":
            return blocked("invalid_real_model_or_past_only_head_training_trajectory")
        seed=r["seed"]
        if seed in results or raw["run_id"] in ids:
            return blocked("duplicate_seed_or_run")
        ids.add(raw["run_id"])
        results[seed]=r
    if set(results)!=set(SEEDS):
        return blocked("incorrect_one_feedback_cohort_inventory")
    episodes=[(seed,ep) for seed in SEEDS for ep in results[seed]["episodes"]]
    policies={}
    for policy in POLICIES:
        deltas=[ep["policies"][policy]["diagnostic_loss_delta"]
                for _,ep in episodes]
        transfer=[ep["policies"][policy]["transfer_loss_delta"]
                  for _,ep in episodes]
        policies[policy]={
            "mean_episode_diagnostic_loss_delta":statistics.mean(deltas),
            "mean_episode_transfer_loss_delta":statistics.mean(transfer),
            "episode_diagnostic_loss_deltas":{
                f"{seed}-{ep['episode']}":ep["policies"][policy]["diagnostic_loss_delta"]
                for seed,ep in episodes},
            "episode_transfer_loss_deltas":{
                f"{seed}-{ep['episode']}":ep["policies"][policy]["transfer_loss_delta"]
                for seed,ep in episodes},
            "temporary_model_diagnostic_steps":sum(
                ep["policies"][policy]["temporary_adapter_step"] for _,ep in episodes),
            "model_training_probe_forwards":len(episodes),
            "model_diagnostic_forwards":len(episodes)*6,
        }
    changes=sum(
        ep["policies"]["adaptive_head_one_probe"]["selected_index"]!=
        ep["policies"]["frozen_head_one_probe"]["selected_index"]
        for _,ep in episodes)
    qk_updates=sum(results[s]["total_online_head_weight_updates"] for s in SEEDS)
    head_objectives=sum(results[s]["total_online_head_only_objectives"] for s in SEEDS)
    if (qk_updates==0 or head_objectives<=0 or
            any(results[s]["frozen_head_unchanged"] is not True or
                results[s]["model_persistent_adapter_updates"]!=0 for s in SEEDS)):
        return blocked("not_truly_trained_without_model_backprop")
    reasons=[]
    if changes==0:
        reasons.append("online_qk_weight_changes_did_not_change_any_of_nine_model_choices")
    adaptive=policies["adaptive_head_one_probe"]
    frozen=policies["frozen_head_one_probe"]
    random=policies["random_one_probe"]
    if adaptive["mean_episode_diagnostic_loss_delta"]>=frozen["mean_episode_diagnostic_loss_delta"]-1e-8:
        reasons.append("adaptive_head_did_not_outperform_frozen_attention_head")
    if adaptive["mean_episode_diagnostic_loss_delta"]>=0:
        reasons.append("no_positive_average_independent_diagnostic_improvement")
    if adaptive["mean_episode_diagnostic_loss_delta"]>=random["mean_episode_diagnostic_loss_delta"]:
        reasons.append("adaptive_head_did_not_outperform_random")
    if any(x>1e-6 for x in adaptive["episode_transfer_loss_deltas"].values()):
        reasons.append("unseen_instruction_family_degradation")
    reasons.append("small_synthetic_task_cohort_and_additional_head_compute_not_matched")
    return {
        "schema":SCHEMA,"status":"hold_not_qualified",
        "blocking_reasons":reasons,
        "seeds":list(SEEDS),
        "training_feedback_labels_per_episode":1,
        "genuine_head_parameter_count":195,
        "pretrained_model_lora_parameter_count":2048,
        "actual_model_loss_forwards":sum(
            results[s]["real_pretrained_model_loss_forwards"] for s in SEEDS),
        "actual_model_prompt_context_forwards":9,
        "actual_online_head_only_objective_forwards":head_objectives,
        "actual_accepted_online_qk_head_updates":qk_updates,
        "episode_choices_changed_by_online_learning":changes,
        "total_policy_episodes":len(episodes),
        "model_candidate_probes_per_policy_per_episode":1,
        "persistent_model_adapter_updates":0,
        "source_execution_cryptographically_attested":False,
        "actual_head_weights_changed_without_model_backprop":True,
        "model_update_effectiveness_improved_proven":False,
        "fully_compute_matched_to_adamw_muon":False,
        "policy_outcomes":policies,
        "model_training_permitted":False,
        "optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
    }
