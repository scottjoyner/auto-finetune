"""Rebuild a real LFM3 new-family task-shift report from training-only evidence.

The source-trained head and task-shift threshold are independently replayed
from fixed historic source receipts. Every new-model prompt context and each
before-probe direction forecast is checked. New task gain labels are absent,
not merely marked hidden. No model loads, gradients or optimizer commands.

Mathematical review does NOT authenticate the original runtime's honesty.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_task_shift_probe import (
    SCHEMA as SOURCE,EPISODES,NEW_SEEDS,OLD_SEEDS,
    curriculum,directions,task_hash,training_source_path,rebuild_training_head,
)
from src.lfm3_dust_text_family_transfer_review import review as verify_source
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.dust-new-family-novelty-review.v1"
HASH=re.compile(r"[0-9a-f]{64}\Z")
ID=re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {
        "schema":SCHEMA,"status":"blocked","reason":reason,
        "head_update_authorized":False,"optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
    }


def _finite(a,b,eps=1e-9):
    return type(a) in (float,int) and type(b) in (float,int) and (
        math.isfinite(a) and math.isfinite(b) and abs(a-b)<=eps)


def review(result,*,repo_root):
    if not isinstance(result,dict):
        return blocked("invalid_runtime_source")
    seed=result.get("new_task_seed")
    if type(seed) is not int or seed not in NEW_SEEDS:
        return blocked("unregistered_new_family")
    old=OLD_SEEDS[NEW_SEEDS.index(seed)]
    expected={
        "schema":SOURCE,"status":"research_unseen_family_novelty_shadow_only",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "source_training_seed":old,"new_task_seed":seed,
        "head_parameters":195,"source_model_selected_adapter_parameters":2048,
        "novel_family":"sort_two_words","new_family_episodes":EPISODES,
        "source_model_prompt_only_forwards":EPISODES,
        "source_model_perturbation_forwards":0,
        "source_model_backward_calls":0,"retained_model_adapter_updates":0,
        "additional_head_parameter_updates":0,"candidate_gain_labels_collected":0,
        "original_selection_partition_opened":False,
        "original_test_partition_opened":False,
        "model_quality_improvement_proven":False,
        "forecast_error_improvement_proven":False,
        "new_family_novelty_detection_only":True,
        "source_runtime_independently_attested":False,
        "optimizer_switch_permitted":False,
        "model_training_permitted":False,
        "weight_promotion_permitted":False,
    }
    if any(result.get(k)!=v for k,v in expected.items()):
        return blocked("invalid_shadow_evidence_or_promotion")
    if (type(result.get("run_id")) is not str or not ID.fullmatch(result["run_id"])
            or type(result.get("adapter_initial_sha256")) is not str
            or not HASH.fullmatch(result["adapter_initial_sha256"])
            or result.get("adapter_final_sha256")!=result["adapter_initial_sha256"]
            or type(result.get("elapsed_s")) not in (int,float)
            or not .00001<=result["elapsed_s"]<=3600):
        return blocked("unverified_runtime_and_adapter")
    path=training_source_path(Path(repo_root),old)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>175000:
        return blocked("unsafe_training_source")
    raw=path.read_bytes()
    if result.get("source_training_receipt_sha256")!=hashlib.sha256(raw).hexdigest():
        return blocked("historic_training_evidence_changed")
    source=json.loads(raw)
    if verify_source(source).get("status")!="manual_transfer_review_only":
        return blocked("unverified_original_trained_head")
    head=rebuild_training_head(source)
    gate=FrozenTaskShiftGate(source=source)
    if (result.get("head_frozen_sha256")!=head.parameter_sha256
            or result.get("head_history_size")!=len(head.history)
            or result.get("frozen_train_only_guard")!=gate.summary()):
        return blocked("trained_head_or_guard_not_reproducible")
    rows=result.get("episodes")
    if not isinstance(rows,list) or len(rows)!=EPISODES:
        return blocked("incomplete_novelty_episode_inventory")
    tasks=curriculum(seed)
    n_ood,n_abstain=0,0
    min_nearest=1.
    for ep,row in enumerate(rows):
        if (not isinstance(row,dict) or row.get("episode")!=ep
                or type(row["episode"]) is not int
                or row.get("family")!="sort_two_words"
                or row.get("prompt_sha256")!=task_hash(tasks[ep])
                or row.get("labeled_gain_observations")!=0
                or row.get("actual_model_context_forwards")!=1
                or row.get("actual_perturbation_forwards")!=0
                or row.get("head_parameter_updates")!=0
                or row.get("model_backward_calls")!=0):
            return blocked("missing_or_tampered_context_episode")
        ctx=row.get("context")
        if (not isinstance(ctx,list) or len(ctx)!=4
                or any(type(x) not in (float,int) or not math.isfinite(x)
                       for x in ctx) or
                not _finite(sum(x*x for x in ctx),1,eps=1e-5)):
            return blocked("invalid_frozen_model_activation")
        cand=row.get("raw_head_predicted_directions")
        if not isinstance(cand,list) or len(cand)!=8:
            return blocked("selectively_reported_or_missing_candidates")
        nearest=max(sum(x*y for x,y in zip(ctx,t))
                    for t in gate.train_contexts)
        min_nearest=min(min_nearest,nearest)
        if nearest<gate.threshold:
            n_ood+=1
        for idx,(cand_idx,sign,d) in enumerate(directions(seed,ep)):
            entry=cand[idx]
            expected_pred=head.predict(d,ctx,episode=12+ep)
            expected_guard=gate.evaluate(ctx,expected_pred["predicted_gain"],
                                         candidate_index=idx)
            if (not isinstance(entry,dict)
                    or entry.get("candidate")!=cand_idx
                    or entry.get("sign")!=sign
                    or type(entry["candidate"]) is not int
                    or type(entry["sign"]) is not int
                    or entry.get("head_keys")!=expected_pred["historical_keys"]
                    or entry.get("actual_perturbation_forward_performed") is not False
                    or entry.get("measured_gain") is not None
                    or not isinstance(entry.get("direction"),list)
                    or len(entry["direction"])!=8
                    or any(not _finite(x,y) for x,y in zip(entry["direction"],d))
                    or not _finite(entry.get("raw_neural_prediction"),
                                   expected_pred["predicted_gain"])):
                return blocked("prediction_not_frozen_or_probe_attestation_false")
            actual=entry.get("guard")
            if not isinstance(actual,dict) or set(actual)!=set(expected_guard):
                return blocked("invalid_guard_result_schema")
            for key,value in expected_guard.items():
                observed=actual[key]
                if type(value) is float:
                    if not _finite(observed,value):
                        return blocked("incorrect_training_only_guard_math")
                elif observed!=value or type(observed) is not type(value):
                    return blocked("incorrect_guard_admission_or_shield_output")
            n_abstain+=int(actual["abstained"])
    return {
        "schema":SCHEMA,"status":"manual_novelty_shadow_review_only",
        "source_training_seed":old,"new_task_seed":seed,
        "family":"sort_two_words","new_family_examples":EPISODES,
        "prompt_only_model_forwards":EPISODES,
        "source_training_context_min_loo":gate.min_prior_train_nearest,
        "frozen_similarity_threshold":gate.threshold,
        "new_family_context_nearest_min":min_nearest,
        "new_family_ood_episode_count":n_ood,
        "candidate_abstentions":n_abstain,
        "model_adapter_modified":False,
        "actual_model_probe_gains_evaluated":False,
        "head_update_authorized":False,
        "source_runtime_independently_attested":False,
        "optimizer_switch_permitted":False,
        "model_training_permitted":False,
        "weight_promotion_permitted":False,
    }
