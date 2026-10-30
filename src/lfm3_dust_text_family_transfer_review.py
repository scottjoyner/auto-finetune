"""Structural replay of frozen LFM3 head on a new public instruction family.

Reconstructs each previously trained Q/K weight, every baseline/neural forecast,
the no-evaluation-feedback boundary and the recorded gains' task-family
provenance. Source LFM3 execution events are unsigned local data: deterministic
replay is not cryptographically independent inference attestation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

from src.lfm3_dust_activation_head import ActivationDirectionHead, _context
from src.lfm3_dust_direction_attention import DirectionAttentionHead
from src.lfm3_dust_online_calibration import CalibratedActivationHead
from src.lfm3_dust_trainable_attention import ForwardOnlyNeuralAttentionHead
from src.lfm3_dust_text_family_curriculum import (
    SEEDS,TRAIN_EPISODES,TRANSFER_EPISODES,curriculum,manifest,
)
from src.lfm3_dust_text_family_transfer_probe import (
    SCHEMA as SOURCE,POLICIES,SIGNED_DIRECTIONS,score_transfer,
)
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.dust-text-transfer-review.v1"
SHA=re.compile(r"[0-9a-f]{64}\Z")
HEX32=re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "neural_head_qualified":False,"optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}


def number(x,low=-100,high=100):
    return type(x) in (float,int) and math.isfinite(x) and low<=x<=high


def _same(a,b,tol=1e-9):
    return number(a,-1e9,1e9) and number(b,-1e9,1e9) and abs(a-b)<=tol


def review(r):
    if not isinstance(r,dict):
        return blocked("not_a_report")
    expected={
        "schema":SOURCE,"status":"research_disjoint_task_family_forecast_only",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "train_episodes":TRAIN_EPISODES,"transfer_episodes":TRANSFER_EPISODES,
        "training_families":["first_word","last_word"],
        "transfer_family":"uppercase","selected_parameter_count":2048,
        "head_parameters":195,"head_model_backward_count":0,
        "model_training_backward_count":0,"model_adapter_update_count":0,
        "source_model_perturbed_forwards":SIGNED_DIRECTIONS*(TRAIN_EPISODES+TRANSFER_EPISODES),
        "source_model_context_forwards":TRAIN_EPISODES+TRANSFER_EPISODES,
        "transfer_family_feedback_used_for_training":False,
        "original_selection_partition_opened":False,
        "original_untouched_test_opened":False,
        "new_real_world_task_quality_proven":False,
        "independent_model_execution_attested":False,
        "optimizer_switch_permitted":False,"model_update_permitted":False,
        "weight_promotion_permitted":False,"trial_authorized":False,
    }
    if any(r.get(k)!=v for k,v in expected.items()):
        return blocked("unapproved_runtime_or_promotion_claim")
    seed=r.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        return blocked("unregistered_seed")
    try:
        if (r.get("source_dataset_manifest")!=manifest(seed)
                or type(r.get("model_run_id")) is not str
                or not HEX32.fullmatch(r["model_run_id"])
                or type(r.get("initial_adapter_sha256")) is not str
                or not SHA.fullmatch(r["initial_adapter_sha256"])
                or r.get("final_adapter_sha256")!=r["initial_adapter_sha256"]
                or not number(r.get("elapsed_s"),1e-9,3600)):
            return blocked("unmatched_model_or_task_provenance")
        events=r.get("events")
        if not isinstance(events,list) or len(events)!=18:
            return blocked("missing_task_family_episodes")
        data=curriculum(seed)
        neural=ForwardOnlyNeuralAttentionHead(seed=seed)
        direction=DirectionAttentionHead()
        activation=ActivationDirectionHead()
        calibrated=CalibratedActivationHead()
        old_gains=[]
        accepted=0
        frozen_sha=None
        frozen_count=None
        for ep,row in enumerate(events):
            train=ep<TRAIN_EPISODES
            phase="train" if train else "transfer_evaluation"
            item=(data["train"][ep] if train
                  else data["transfer_evaluation"][ep-TRAIN_EPISODES])
            h=hashlib.sha256(json.dumps({k:v for k,v in item.items() if k!="input_key"},
                                         sort_keys=True,separators=(",",":")).encode()).hexdigest()
            if (not isinstance(row,dict) or row.get("episode")!=ep
                    or type(row["episode"]) is not int
                    or row.get("phase")!=phase
                    or row.get("family")!=item["family"]
                    or row.get("example_sha256")!=h
                    or row.get("perturbed_forwards")!=SIGNED_DIRECTIONS
                    or row.get("context_only_forwards")!=1
                    or row.get("gradient_backwards")!=0
                    or not number(row.get("clean_loss"),1e-9,1000)):
                return blocked("incorrect_task_split_or_runtime_event")
            ctx=row.get("context")
            if (not isinstance(ctx,list) or len(ctx)!=4
                    or any(not number(v,-1,1) for v in ctx)
                    or not _same(sum(x*x for x in ctx),1.0,tol=1e-5)
                    or hashlib.sha256(json.dumps(tuple(ctx),separators=(",",":")).encode()).hexdigest()
                       !=row.get("context_sha256")):
                return blocked("invalid_prompt_only_activation_context")
            context=_context(ctx)
            if train or ep==TRAIN_EPISODES:
                head_train=neural.tune_from_previous(episode=ep)
                if row.get("head_training_before_forecast")!=head_train:
                    return blocked("nonreproducible_past_only_head_training")
                accepted+=head_train["head_updates_accepted"]
                if ep==TRAIN_EPISODES:
                    frozen_sha=neural.parameter_sha256
                    frozen_count=len(neural.history)
            elif row.get("head_training_before_forecast") is not None:
                return blocked("transfer_family_trained_neural_head")
            if not train and (neural.parameter_sha256!=frozen_sha
                              or len(neural.history)!=frozen_count):
                return blocked("unfrozen_neural_head_during_transfer")
            candidates=row.get("signed_directions")
            if not isinstance(candidates,list) or len(candidates)!=SIGNED_DIRECTIONS:
                return blocked("selective_sampling_or_incomplete_directions")
            old_mean=statistics.mean(old_gains) if old_gains else 0.
            training_rows=[]
            for i,c in enumerate(candidates):
                sign=1 if i%2==0 else -1
                if (not isinstance(c,dict) or c.get("candidate_index")!=i//2
                        or type(c["candidate_index"]) is not int
                        or c.get("sign")!=sign or type(c["sign"]) is not int):
                    return blocked("wrong_antithetic_direction_index")
                v=c.get("direction")
                if (not isinstance(v,list) or len(v)!=8
                        or any(not number(x,-2,2) for x in v)
                        or not _same(sum(x*x for x in v),1.,tol=1e-5)):
                    return blocked("bad_direction_coefficients")
                if i%2 and any(not _same(x,-y,tol=1e-7)
                               for x,y in zip(v,candidates[i-1]["direction"],strict=True)):
                    return blocked("unmatched_plus_minus_direction")
                if not number(c.get("measured_gain")):
                    return blocked("bad_measured_forward_gain")
                p={
                    "zero":0.,
                    "historical_mean":old_mean,
                    "direction":direction.predict(v,episode=ep)["predicted_gain"],
                    "activation":activation.predict(v,context,episode=ep)["predicted_gain"],
                    "calibrated":calibrated.predict(v,context,episode=ep)["predicted_gain"],
                    "neural":neural.predict(v,context,episode=ep)["predicted_gain"],
                }
                if (not isinstance(c.get("predictions"),dict)
                        or set(c["predictions"])!=set(POLICIES)
                        or any(not _same(c["predictions"][name],z)
                               for name,z in p.items())
                        or c.get("neural_prior_attention_keys")!=
                           neural.predict(v,context,episode=ep)["historical_keys"]
                        or c.get("prior_direction_attention_keys")!=
                           direction.predict(v,episode=ep)["evidence"]):
                    return blocked("hindsight_forecast_or_wrong_neural_weights")
                training_rows.append((v,context,c["measured_gain"]))
            if train:
                for v,ctx,gain in training_rows:
                    direction.observe(v,gain,episode=ep,split="train",probed=True)
                    activation.observe(v,ctx,gain,episode=ep,split="train",measured=True)
                    calibrated.observe(v,ctx,gain,episode=ep,split="train",measured=True)
                    old_gains.append(gain)
                neural.observe_episode(training_rows,episode=ep,split="train",measured=True)
            if (row.get("neural_head_sha_after_episode")!=neural.parameter_sha256
                    or row.get("neural_training_memory_after_episode")!=len(neural.history)):
                return blocked("neural_memory_or_parameters_mutated")
        if (r.get("head_frozen_before_transfer_sha256")!=frozen_sha
                or r.get("head_final_sha256")!=frozen_sha
                or r.get("head_frozen_before_transfer_memory_size")!=frozen_count
                or r.get("head_parameter_updates_count")!=accepted
                or r.get("head_only_loss_evaluations")!=neural.trial_head_forwards):
            return blocked("incorrect_head_freeze_or_training_cost")
        exact=score_transfer(events)
        reported=r.get("transfer_metrics")
        if not isinstance(reported,dict) or set(reported)!=set(POLICIES):
            return blocked("incomplete_transfer_metrics")
        for name in POLICIES:
            if not isinstance(reported[name],dict) or set(reported[name])!=set(exact[name]):
                return blocked("unexpected_metric_schema")
            if any(not _same(reported[name][key],value)
                   for key,value in exact[name].items()):
                return blocked("misreported_transfer_score")
        zero,learned=exact["zero"],exact["neural"]
        if zero["mean_squared_error"]<=0:
            return blocked("unusable_zero_gain_control")
        return {
            "schema":SCHEMA,"status":"manual_transfer_review_only",
            "seed":seed,"train_prompts":TRAIN_EPISODES,
            "distinct_transfer_family":"uppercase",
            "transfer_prompts":TRANSFER_EPISODES,
            "neural_head_updated":accepted>0,
            "accepted_head_only_updates":accepted,
            "head_only_loss_evaluations":neural.trial_head_forwards,
            "transfer_cases":learned["prospective_cases"],
            "neural_mse":learned["mean_squared_error"],
            "zero_mse":zero["mean_squared_error"],
            "neural_mse_ratio":learned["mean_squared_error"]/zero["mean_squared_error"],
            "neural_top1_gain_advantage":(
                learned["mean_gain_if_top1_tested"]-zero["mean_gain_if_top1_tested"]),
            "model_adapter_unchanged":True,"evaluation_feedback_never_used":True,
            "independent_runtime_attestation":False,
            "optimizer_switch_permitted":False,"model_update_permitted":False,
            "weight_promotion_permitted":False,
        }
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ZeroDivisionError):
        return blocked("unreconstructible_historical_or_transfer_evidence")
