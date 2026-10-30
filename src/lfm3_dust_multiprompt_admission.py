"""Three-seed admission of new Dust multi-prompt LoRA-B quality evidence.

The gate must BOTH avoid harmful diagnostic updates and retain useful ones
across independent prompts before even nominating a fresh confirmation study.
Nothing in this file authorizes live model training or a checkpoint promotion.
"""
from __future__ import annotations

from src.lfm3_dust_multiprompt_review import review

SCHEMA="sensorgraft.lfm3.dust-multiprompt-admission.v1"
SEEDS=(19,20,21)


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}


def assess(reports,*,repo_root):
    if not isinstance(reports,list) or len(reports)!=3:
        return blocked("three_new_pretrained_runs_required")
    checked={}
    ids=set()
    for r in reports:
        v=review(r,repo_root=repo_root)
        if v.get("status")!="manual_multiprompt_quality_review_only":
            return blocked("source_model_or_head_predictions_not_reproducible")
        seed=v["seed"]
        if seed in checked or r["run_id"] in ids:
            return blocked("repeated_seed_or_run")
        checked[seed]=v
        ids.add(r["run_id"])
    if set(checked)!=set(SEEDS):
        return blocked("wrong_new_seed_inventory")
    policies=("random","neural","greedy")
    means={}
    harmful={}
    false_veto={}
    for policy in policies:
        single=[checked[s]["results"][f"{policy}_single"]["diagnostic_loss_change"]
                for s in SEEDS]
        multi=[checked[s]["results"][f"{policy}_multi"]["diagnostic_loss_change"]
               for s in SEEDS]
        means[policy]={"single_mean_loss_delta":sum(single)/3,
                       "multi_mean_loss_delta":sum(multi)/3,
                       "per_seed_single":dict(zip(map(str,SEEDS),single,strict=True)),
                       "per_seed_multi":dict(zip(map(str,SEEDS),multi,strict=True))}
        harmful[policy]=sum(x>1e-8 for x in multi)
        false_veto[policy]=sum(
            checked[s]["results"][f"{policy}_single"]["applied"] and
            not checked[s]["results"][f"{policy}_multi"]["applied"] and
            checked[s]["results"][f"{policy}_single"]["diagnostic_loss_change"] < -1e-8
            for s in SEEDS
        )
    reasons=[]
    if any(harmful.values()):
        reasons.append("three_prompt_guard_still_accepts_harmful_diagnostic_update")
    if any(false_veto.values()):
        reasons.append("three_prompt_guard_rejects_useful_diagnostic_update")
    if not all(means[p]["multi_mean_loss_delta"]<0 for p in ("neural","greedy")):
        reasons.append("three_prompt_neural_or_greedy_does_not_improve_diagnostics")
    return {
        "schema":SCHEMA,
        "status":"hold_not_qualified" if reasons else "separate_confirmatory_study_only",
        "blocking_reasons":reasons,
        "seeds":list(SEEDS),
        "training_examples_per_seed":3,
        "diagnostic_examples_per_seed":6,
        "total_real_model_loss_forwards":sum(checked[s]["total_real_model_loss_forwards"]
                                             for s in SEEDS),
        "total_new_context_forwards":3,
        "total_model_backwards":0,"persistent_adapter_updates":0,
        "policy_diagnostic_deltas":means,
        "harmful_multi_prompt_updates":harmful,
        "beneficial_single_prompt_updates_vetoed":false_veto,
        "models_or_adapters_promoted":0,
        "independent_execution_attested":False,
        "comparison_with_adamw_muon_compute_matched":False,
        "optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }
