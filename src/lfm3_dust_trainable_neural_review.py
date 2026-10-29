"""Fail-closed independent replay of real LFM3 neural attention-head receipts.

Actual frozen-model forwards are checked by the preceding all-candidate source
reviewer; this second verifier deterministically reconstructs *all* 195 neural
parameters, previous-episode antithetic head-only updates and each prospective
forecast from serialized prompt-activation summaries and measured train gains.

This is not a source-independent model-execution attestation or proof of
downstream task-quality improvement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from src.lfm3_dust_prequential_review import review as review_base
from src.lfm3_dust_prequential_audit import _metrics, NEURAL_POLICIES
from src.lfm3_dust_trainable_attention import (
    ForwardOnlyNeuralAttentionHead, HEAD_PARAM_COUNT, _context, _hash,
)

SCHEMA="sensorgraft.lfm3.dust-trainable-neural-review.v1"
NEW_SEEDS=(7,8,9)
NEURAL_METADATA={
    "neural_head","neural_head_parameters","neural_head_training",
    "neural_head_final_sha256","neural_head_total_only_forward_evaluations",
    "neural_head_model_backpropagation","neural_head_optimizer_object",
    "head_parameter_training_observed",
    "neural_head_transformed_predictions_are_not_production_authorized",
}


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "head_policy_qualified":False,"optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}


def _number(x):
    return type(x) in (int,float) and math.isfinite(x)


def review(receipt):
    if not isinstance(receipt,dict):
        return blocked("not_a_report")
    seed=receipt.get("seed")
    if type(seed) is not int or seed not in NEW_SEEDS or receipt.get("episodes")!=12:
        return blocked("wrong_frozen_neural_cohort")
    if (receipt.get("neural_head")!="trainable_two_head_scaled_dot_product_cross_attention"
            or receipt.get("neural_head_parameters")!=HEAD_PARAM_COUNT
            or receipt.get("neural_head_training")!="two_forward_only_antithetic_head_trials_per_episode"
            or receipt.get("neural_head_model_backpropagation") is not False
            or receipt.get("neural_head_optimizer_object") is not False
            or receipt.get("neural_head_transformed_predictions_are_not_production_authorized") is not True):
        return blocked("unauthorized_neural_controller_contract")
    rows=receipt.get("episode_receipts")
    if not isinstance(rows,list) or len(rows)!=12:
        return blocked("missing_neural_episode_ledger")
    # The predecessor reviewer still checks every real-model +/- candidate,
    # frozen adapter checksum, training task partition and baseline forecast.
    ordinary=copy.deepcopy(receipt)
    for key in NEURAL_METADATA:
        if key not in ordinary:
            return blocked("missing_neural_provenance")
        ordinary.pop(key)
    if set(receipt)-set(ordinary)!=NEURAL_METADATA:
        return blocked("neural_metadata_mismatch")
    ordinary.get("metrics",{}).pop("neural",None)
    if not isinstance(ordinary.get("metrics"),dict) or set(ordinary["metrics"])!=set(NEURAL_POLICIES)-{"neural"}:
        return blocked("missing_prior_baselines")
    for row in ordinary["episode_receipts"]:
        for key in ("preprobe_neural_context","neural_training",
                    "neural_head_after_episode_sha256","neural_head_parameter_count"):
            row.pop(key,None)
        for candidate in row.get("candidates",[]):
            candidate.pop("neural_historical_keys",None)
            if isinstance(candidate.get("predictions"),dict):
                candidate["predictions"].pop("neural",None)
    base=review_base(ordinary)
    if base.get("status")!="descriptive_review_required":
        return blocked("source_model_trial_failed_"+base.get("reason","unknown"))

    head=ForwardOnlyNeuralAttentionHead(seed=seed)
    initial=_hash(head.initial)
    total_head_evals=0
    forecasts=[]
    for ep,row in enumerate(rows):
        training=head.tune_from_previous(episode=ep)
        logged=row.get("neural_training")
        if logged!=training:
            return blocked("unreproducible_previous_episode_head_training")
        total_head_evals+=training["head_only_forward_evaluations"]
        context=row.get("preprobe_neural_context")
        try:
            normalized=_context(context)
        except (ValueError,TypeError):
            return blocked("invalid_prompt_attention_input")
        if (len(context)!=4 or
                hashlib.sha256(json.dumps(normalized,separators=(",",":")).encode()).hexdigest()!=row["context_sha256"]):
            return blocked("unverified_preprobe_model_context")
        if row.get("neural_head_parameter_count")!=HEAD_PARAM_COUNT:
            return blocked("wrong_neural_head_parameter_count")
        gains=[]
        for entry in row["candidates"]:
            pred=head.predict(entry["direction_coefficients"],normalized,episode=ep)
            score=entry.get("predictions",{}).get("neural")
            if (not _number(score) or abs(score-pred["predicted_gain"])>1e-9
                    or entry.get("neural_historical_keys")!=pred["historical_keys"]):
                return blocked("neural_forecast_contains_hindsight_or_wrong_parameters")
            gains.append((entry["direction_coefficients"],normalized,entry["measured_gain"]))
            forecasts.append(score)
        head.observe_episode(gains,episode=ep,split="train",measured=True)
        if row.get("neural_head_after_episode_sha256")!=head.parameter_sha256:
            return blocked("missing_or_forged_trained_head_checksum")
    if (receipt.get("neural_head_final_sha256")!=head.parameter_sha256
            or receipt.get("neural_head_total_only_forward_evaluations")!=total_head_evals
            or receipt.get("head_parameter_training_observed") is not (initial!=head.parameter_sha256)):
        return blocked("head_training_ledger_not_matching_frozen_replay")

    try:
        recomputed=_metrics(rows,expected_episodes=12,policies=NEURAL_POLICIES)
    except (KeyError,ValueError,OverflowError,TypeError):
        return blocked("unreconstructible_neural_prequential_metrics")
    metrics=receipt.get("metrics")
    if not isinstance(metrics,dict) or set(metrics)!=set(NEURAL_POLICIES):
        return blocked("policy_metric_inventory_invalid")
    for policy in NEURAL_POLICIES:
        if not isinstance(metrics[policy],dict) or set(metrics[policy])!=set(recomputed[policy]):
            return blocked("metric_schema_mismatch")
        if any(not _number(metrics[policy][k]) or
               abs(metrics[policy][k]-v)>1e-9
               for k,v in recomputed[policy].items()):
            return blocked("metric_recomputed_differently")
    zero,neural=recomputed["zero"],recomputed["neural"]
    return {
        "schema":SCHEMA,
        "status":"manual_neural_head_review_only",
        "seed":seed,"head_trainable_params":HEAD_PARAM_COUNT,
        "head_updated_from_prior_forward_only_feedback":initial!=head.parameter_sha256,
        "total_head_only_loss_evaluations":total_head_evals,
        "new_model_backward_calls":0,
        "model_adapter_unchanged":True,
        "prequential_neural_mse":neural["mse"],
        "prequential_zero_mse":zero["mse"],
        "neural_mse_ratio_vs_zero":neural["mse"]/zero["mse"],
        "neural_top1_gain_advantage_vs_zero":(
            neural["mean_gain_if_top1_probed"]-zero["mean_gain_if_top1_probed"]),
        "prospective_cases":neural["cases"],
        "new_task_family_generalization_proven":False,
        "source_independently_authenticated":False,
        "head_policy_qualified":False,"optimizer_switch_permitted":False,
        "model_update_permitted":False,"weight_promotion_permitted":False,
    }
