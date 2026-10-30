"""Prospective three-seed decision for variance-aware forward-only Dust.

No claim of optimizer qualification can follow from mean improvement over
three synthetic seeds. A useful rule needs credible benefits without
withholding other useful updates and must beat cheaper no-update/random
baselines on a large independent evaluation. This pilot always stays HOLD.
"""
from __future__ import annotations
import statistics
from src.lfm3_dust_variance_review import review

SCHEMA="sensorgraft.lfm3.dust-variance-admission.v1"
SEEDS=(22,23,24)
METHODS=("random","neural","greedy")
MODES=("single","multi","variance")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,"model_update_permitted":False,
            "weight_promotion_permitted":False}


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=3:
        return blocked("three_new_pretrained_variance_cohorts_required")
    seen={}
    run_ids=set()
    for source in reports:
        r=review(source,repo_root=repo_root)
        if r.get("status")!="manual_multiprompt_quality_review_only":
            return blocked("failed_exact_neural_head_training_or_quality_replay")
        seed=r["seed"]
        if seed in seen or source["run_id"] in run_ids:
            return blocked("duplicate_seed_or_run")
        seen[seed]=r
        run_ids.add(source["run_id"])
    if set(seen)!=set(SEEDS):
        return blocked("wrong_research_seed_inventory")
    metrics={}
    for policy in METHODS:
        by_mode={}
        for mode in MODES:
            rows=[seen[s]["results"][f"{policy}_{mode}"] for s in SEEDS]
            deltas=[r["diagnostic_loss_change"] for r in rows]
            by_mode[mode]={
                "per_seed_diagnostic_loss_delta":{str(s):v for s,v in zip(SEEDS,deltas,strict=True)},
                "mean_seedwise_diagnostic_loss_delta":statistics.mean(deltas),
                "temporary_steps_applied":sum(r["applied"] for r in rows),
                "harmful_temporary_steps_applied":sum(r["applied"] and
                    r["diagnostic_loss_change"]>1e-8 for r in rows),
            }
        by_mode["useful_single_updates_vetoed_by_variance"]=sum(
            seen[s]["results"][f"{policy}_single"]["applied"] and
            not seen[s]["results"][f"{policy}_variance"]["applied"] and
            seen[s]["results"][f"{policy}_single"]["diagnostic_loss_change"] < -1e-8
            for s in SEEDS)
        metrics[policy]=by_mode
    reasons=[]
    neural=metrics["neural"]
    if neural["useful_single_updates_vetoed_by_variance"]>0:
        reasons.append("variance_gate_rejects_useful_head_selected_update")
    if neural["variance"]["harmful_temporary_steps_applied"]>0:
        reasons.append("variance_gate_does_not_prevent_harmful_updates")
    if neural["variance"]["temporary_steps_applied"]<2:
        reasons.append("not_enough_independent_accepted_neural_updates")
    if not all(neural["variance"]["per_seed_diagnostic_loss_delta"][str(s)]<0 for s in SEEDS):
        reasons.append("neural_update_not_consistently_better_than_no_update")
    if neural["variance"]["mean_seedwise_diagnostic_loss_delta"]>=metrics["random"]["variance"]["mean_seedwise_diagnostic_loss_delta"]:
        reasons.append("neural_variance_not_better_than_random_variance")
    return {
        "schema":SCHEMA,"status":"hold_not_qualified",
        "blocking_reasons":reasons,
        "seeds":list(SEEDS),
        "trained_neural_head_parameters":195,
        "lora_b_parameter_scope":2048,
        "real_model_loss_forwards":sum(seen[s]["total_real_model_loss_forwards"] for s in SEEDS),
        "real_prompt_activation_forwards":3,
        "lora_b_temporary_weights_restored":True,
        "persistent_adapter_updates":0,
        "model_backpropagation_calls":0,
        "exact_3_prompt_variance_proxy_not_confidence_interval":True,
        "head_learned_from_new_quality_labels":False,
        "source_runtime_cryptographically_attested":False,
        "methods":metrics,
        "compute_matched_adamw_muon_dust":False,
        "optimizer_switch_permitted":False,"model_update_permitted":False,
        "weight_promotion_permitted":False,
    }
