"""Fail-closed cross-format admission for LFM3 native-optimizer advice.

This policy is deliberately a *last-mile research reviewer*, not a training
executor.  Four original A votes alone cannot authorize a suggestion: the
original 1.7B model must also choose B with the same losses label-swapped.
Even 8/8 does not confer source authentication, heldout quality or execution.
"""
from __future__ import annotations
import math

from src.lfm3_ffn_schedule_counterbalance import (
    SCHEMA, SOURCE_OBSERVER, counterbalance_judge
)
from src.lfm3_ffn_schedule_self_observer import prepare_views
from src.lfm3_optimizer_guard import OptimizerResearchRefused

FIELDS={
    "schema","checkpoint_sha256","original_model_votes","reversed_model_votes",
    "original_format_all_correct","reversed_format_all_correct",
    "correct_votes_out_of_eight",
    "counterbalanced_research_evidence_sufficient_for_execution",
    "advice_reliable_across_all_eight_tested_views","model_can_select_new_lr",
    "model_can_train","model_can_promote","independently_authenticated",
    "heldout_evaluated","optimizer_modified","weights_modified","production_ready",
    "experiment_date","node","source_commit","network_disabled",
    "original_model_unmodified","holdout_custody_verified",
    "label_bias_experiment_only","source_observer_commit",
}

def evaluate_optimizer_advisor(calibration:object, original_development:object,
                               schedule:object, original_observation:object,
                               counterbalanced_receipt:object) -> dict:
    _, numeric=prepare_views(schedule,calibration,original_development)
    receipt=counterbalanced_receipt
    if (type(receipt) is not dict or set(receipt)!=FIELDS
            or receipt["schema"]!=SCHEMA
            or receipt["checkpoint_sha256"]!=numeric["checkpoint_sha256"]
            or receipt["source_observer_commit"]!=SOURCE_OBSERVER
            or receipt["source_commit"]!="20c62b744cb7a56ca29a4ebca2f39ae34aaf7075"
            or receipt["experiment_date"]!="2026-10-28"
            or receipt["node"]!="xwing"
            or receipt["network_disabled"] is not True
            or receipt["original_model_unmodified"] is not True
            or receipt["holdout_custody_verified"] is not False
            or receipt["label_bias_experiment_only"] is not True
            or receipt["counterbalanced_research_evidence_sufficient_for_execution"] is not False
            or receipt["model_can_select_new_lr"] is not False
            or receipt["model_can_train"] is not False
            or receipt["model_can_promote"] is not False
            or receipt["independently_authenticated"] is not False
            or receipt["heldout_evaluated"] is not False
            or receipt["optimizer_modified"] is not False
            or receipt["weights_modified"] is not False
            or receipt["production_ready"] is not False):
        raise OptimizerResearchRefused("untrusted counterbalanced optimizer advice")
    checked=counterbalance_judge(
        numeric,original_observation,receipt["reversed_model_votes"]
    )
    fields=(
        "original_model_votes","reversed_model_votes",
        "original_format_all_correct","reversed_format_all_correct",
        "correct_votes_out_of_eight",
        "counterbalanced_research_evidence_sufficient_for_execution",
        "advice_reliable_across_all_eight_tested_views",
        "model_can_select_new_lr","model_can_train","model_can_promote",
        "independently_authenticated","heldout_evaluated","optimizer_modified",
        "weights_modified","production_ready"
    )
    if any(checked[k]!=receipt[k] for k in fields):
        raise OptimizerResearchRefused("observer evidence does not match re-computed gate")
    count=checked["correct_votes_out_of_eight"]
    if type(count) is not int or count not in range(5,9) and count!=4:
        raise OptimizerResearchRefused("invalid original-model counterbalance result")
    return {
        "schema":"sg-lfm3-native-optimizer-advisor-admission.v1",
        "model_checkpoint_sha256":numeric["checkpoint_sha256"],
        "original_model_comparisons_correct":4,
        "reversed_model_comparisons_correct":count-4,
        "paired_observation_accuracy":count/8,
        "label_position_bias_detected":not checked["reversed_format_all_correct"],
        "model_advisor_reliable_for_this_fixed_comparison":
            checked["reversed_format_all_correct"],
        # A genuine real model failure has occurred. No new 'model-directed'
        # optimizer experiments can be admitted on the same spent evidence.
        "reviewable_hypothesis":(
            {"optimizer":"adamw","learning_rates":[0.004]*3,
             "execute":False,"requires_fresh_independent_tasks":True}
            if checked["reversed_format_all_correct"] else None
        ),
        "counterbalanced_evidence_authentic":False,
        "heldout_evaluated":False,
        "model_may_execute":False,
        "model_may_change_optimizer":False,
        "model_may_promote":False,
        "production_ready":False,
    }
