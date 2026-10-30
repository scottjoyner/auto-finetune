"""Real-data read-only bridge: a trained LFM3 neural head to Dust shadow order.

Uses an ALREADY committed pinned-model receipt. Head Q/K weights are rebuilt
from past TRAINING prompts, verified against the freeze hash, then asked to
rank the eight frozen BEFORE-probe candidate directions from a distinct
UPPERCASE evaluation prompt. This replays preprobe scores only and NEVER
updates an LFM3 adapter, issues a model probe, or uses evaluation labels
for head learning.

The admission veto is mandatory; no caller can turn this into a live policy.
"""
from __future__ import annotations

import math

from src.lfm3_dust_trainable_attention import ForwardOnlyNeuralAttentionHead
from src.lfm3_dust_text_family_transfer_review import review as verify_transfer
from src.lfm3_three_method_shadow_registry import shadow_rank, blocked


def replay_frozen_neural_shadow(receipt,admission,*,transfer_index=0):
    if (type(transfer_index) is not int or not 0<=transfer_index<6
            or not isinstance(receipt,dict) or not isinstance(admission,dict)
            or admission.get("status")!="hold_not_qualified"
            or admission.get("predictor_qualified_for_model_updates") is not False):
        return blocked("unqualified_head_or_invalid_transfer_index")
    evidence=verify_transfer(receipt)
    if evidence.get("status")!="manual_transfer_review_only":
        return blocked("failed_frozen_neural_transfer_review")
    head=ForwardOnlyNeuralAttentionHead(seed=receipt["seed"])
    for ep in range(12):
        log=head.tune_from_previous(episode=ep)
        row=receipt["events"][ep]
        if log!=row["head_training_before_forecast"]:
            return blocked("replay_training_event_changed")
        c=row["context"]
        result=head.observe_episode([
            (entry["direction"],c,entry["measured_gain"])
            for entry in row["signed_directions"]],
            episode=ep,split="train",measured=True)
        if result["head_weights_sha256"]!=row["neural_head_sha_after_episode"]:
            return blocked("neural_training_hash_disagreement")
    head.tune_from_previous(episode=12)
    if head.parameter_sha256!=receipt["head_frozen_before_transfer_sha256"]:
        return blocked("unverifiable_head_freeze")
    row=receipt["events"][12+transfer_index]
    candidate_directions=[c["direction"] for c in row["signed_directions"]]
    result=shadow_rank(
        head,row["context"],candidate_directions,
        episode=12+transfer_index,admission=admission)
    if result.get("status")!="predictive_head_shadow_abstention":
        return blocked("neural_head_not_in_shadow_mode")
    if any(not math.isclose(result["candidates"][i]["predicted_gain"],
                            c["predictions"]["neural"],
                            rel_tol=0,abs_tol=1e-9)
           for i,c in enumerate(row["signed_directions"])):
        return blocked("preprobe_shadow_predictions_mismatch")
    return {
        **result,
        "schema":"sensorgraft.lfm3.real-neural-head-shadow-bridge.v1",
        "seed":receipt["seed"],
        "phase":"frozen_unseen_task_research_replay",
        "episode":12+transfer_index,
        "actual_head_parameter_count":195,
        "head_parameter_sha256":head.parameter_sha256,
        "training_label_epochs_used":12,
        "unseen_task_family":row["family"],
        "evaluated_using_preprobe_forecasts_only":True,
        "actual_adapter_updates_performed":0,
        "additional_model_forwards_performed":0,
        "independent_runtime_attestation":False,
        "head_policy_qualified":False,
        "selected_for_live_probe":None,
        "model_training_permitted":False,
        "weight_promotion_permitted":False,
    }
