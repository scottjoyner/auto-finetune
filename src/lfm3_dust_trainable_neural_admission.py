"""Strict 3-seed decision gate for derivative-free trainable LFM3 predictor.

Passing merely nominates a *new* independent real-task study; it NEVER
authorizes optimizer changes, model training, deployment or weight promotion.
"""
from __future__ import annotations
import statistics

from src.lfm3_dust_trainable_neural_review import review
from src.lfm3_lora_text_task_probe import dataset

SCHEMA="sensorgraft.lfm3.trainable-head-admission.v1"
SEEDS=(7,8,9)


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "head_policy_qualified":False,"optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}


def assess(sources):
    if not isinstance(sources,list) or len(sources)!=3:
        return blocked("three_real_model_receipts_required")
    found={}
    runs=set()
    for receipt in sources:
        r=review(receipt)
        if r.get("status")!="manual_neural_head_review_only":
            return blocked("unverified_real_model_or_neural_head_training")
        seed=r["seed"]
        if seed in found or receipt["model_run_id"] in runs:
            return blocked("duplicate_seed_or_run")
        found[seed]=r
        runs.add(receipt["model_run_id"])
    if set(found)!=set(SEEDS):
        return blocked("wrong_confirmatory_seed_inventory")
    examples=[row["input_key"] for seed in SEEDS for row in dataset(seed)["train"][:12]]
    unique=len(set(examples))
    if unique<32:
        return blocked("insufficient_unique_training_examples")
    ratios=[found[seed]["neural_mse_ratio_vs_zero"] for seed in SEEDS]
    gains=[found[seed]["neural_top1_gain_advantage_vs_zero"] for seed in SEEDS]
    all_mse=all(x<=.95 for x in ratios)
    all_gain=all(x>0 for x in gains)
    weights=all(found[s]["head_updated_from_prior_forward_only_feedback"] for s in SEEDS)
    holds=[]
    if not all_mse: holds.append("forecast_error_not_5pct_below_zero_on_all_seeds")
    if not all_gain: holds.append("top1_gain_not_positive_on_all_seeds")
    if not weights: holds.append("no_verified_head_training_on_every_seed")
    return {"schema":SCHEMA,
            "status":"hold_not_qualified" if holds else "separate_confirmation_study_only",
            "blocking_reasons":holds,"seeds":list(SEEDS),
            "unique_training_task_examples":unique,
            "prospective_forecasts_per_method":sum(found[s]["prospective_cases"] for s in SEEDS),
            "mean_seedwise_ratio_vs_zero":statistics.mean(ratios),
            "every_seed_forecast_mse_5pct_below_zero":all_mse,
            "every_seed_top1_gain_advantage_positive":all_gain,
            "head_parameters_trained_without_backprop":weights,
            "per_seed":{str(s):found[s] for s in SEEDS},
            "head_policy_qualified":False,"optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}
