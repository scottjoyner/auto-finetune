"""Conservative task-family transfer admission for trained LFM3 Dust heads.

The head is fit to first/last word extraction and evaluated with its weights
and memory frozen on uppercase rewriting. A repeatable win vs zero gain and
positive top-1 acquisition on *each seed* is needed even to nominate a fresh
prospective study. No result ever authorizes deployment or model updates.
"""
from __future__ import annotations
import statistics

from src.lfm3_dust_text_family_transfer_review import review
from src.lfm3_dust_text_family_curriculum import SEEDS,curriculum

SCHEMA="sensorgraft.lfm3.dust-text-transfer-admission.v1"


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "predictor_qualified_for_model_updates":False,
            "optimizer_switch_permitted":False,"model_update_permitted":False,
            "weight_promotion_permitted":False}


def assess(sources):
    if not isinstance(sources,list) or len(sources)!=len(SEEDS):
        return blocked("exact_three_seed_receipts_required")
    seen=set()
    receipts={}
    for report in sources:
        result=review(report)
        if result.get("status")!="manual_transfer_review_only":
            return blocked("invalid_real_model_or_neural_feedback_receipt")
        seed=result["seed"]
        if seed in receipts or report["model_run_id"] in seen:
            return blocked("duplicated_seed_or_execution_identity")
        seen.add(report["model_run_id"])
        receipts[seed]=result
    if set(receipts)!=set(SEEDS):
        return blocked("invalid_transfer_seed_inventory")
    training_ids=[r["input_key"] for seed in SEEDS for r in curriculum(seed)["train"]]
    transfer_ids=[r["input_key"] for seed in SEEDS for r in curriculum(seed)["transfer_evaluation"]]
    if len(set(training_ids))!=36 or len(set(transfer_ids))!=18:
        return blocked("duplicate_task_examples_not_independent")
    ratios=[receipts[seed]["neural_mse_ratio"] for seed in SEEDS]
    gain=[receipts[seed]["neural_top1_gain_advantage"] for seed in SEEDS]
    all_mse=all(x<=.95 for x in ratios)
    all_gain=all(x>0 for x in gain)
    all_learned=all(receipts[seed]["neural_head_updated"] for seed in SEEDS)
    reasons=[]
    if not all_mse:
        reasons.append("neural_transfer_forecast_not_5pct_better_than_zero_each_seed")
    if not all_gain:
        reasons.append("top1_transfer_gain_not_positive_each_seed")
    if not all_learned:
        reasons.append("actual_neural_head_updates_not_observed")
    return {
        "schema":SCHEMA,
        "status":"hold_not_qualified" if reasons else "separate_confirmation_study_only",
        "blocking_reasons":reasons,
        "source_seeds":list(SEEDS),
        "training_examples_unique":36,
        "unseen_instruction_examples_unique":18,
        "transfer_signed_direction_forecasts_per_policy":144,
        "mean_seedwise_neural_mse_ratio_vs_zero":statistics.mean(ratios),
        "all_seed_neural_mse_beats_zero_5pct":all_mse,
        "all_seed_neural_top1_gain_advantage_positive":all_gain,
        "head_trained_without_backprop":all_learned,
        "per_seed":{str(s):receipts[s] for s in SEEDS},
        "real_world_generalization_proven":False,
        "predictor_qualified_for_model_updates":False,
        "optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }
