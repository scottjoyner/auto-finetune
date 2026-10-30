"""Frozen LFM3-1.7B task-family-transfer audit of forward-only neural attention.

Explicit, offline CPU research, no automatic adapter updates. Train neural
attention Q/K weights on 12 public FIRST/LAST extraction prompts via only
historical scalar forward-probe gains, then freeze all head weights/memory
before evaluating 6 public UPPERCASE prompts. The transfer-family measured
gains are used ONLY to score prior predictions. Every candidate is measured
regardless of its prediction, controlling selective-probe bias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from src.lfm3_dust_activation_head import (
    ActivationDirectionHead, prompt_only_activation_context,
)
from src.lfm3_dust_direction_attention import DIM, DirectionAttentionHead
from src.lfm3_dust_online_calibration import CalibratedActivationHead
from src.lfm3_dust_trainable_attention import ForwardOnlyNeuralAttentionHead
from src.lfm3_dust_predictive_probe import _basis, _direction, _loss, _unit
from src.lfm3_dust_text_family_curriculum import (
    SEEDS, TRAIN_EPISODES, TRANSFER_EPISODES, curriculum, manifest,
)
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA="sensorgraft.lfm3.dust-neural-head-text-family-transfer.v1"
SIGNED_DIRECTIONS=8
SIGMA=.05
POLICIES=("zero","historical_mean","direction","activation","calibrated","neural")


def validate(*,seed,train_episodes=TRAIN_EPISODES,transfer_episodes=TRANSFER_EPISODES,
             signed_directions=SIGNED_DIRECTIONS,sigma=SIGMA):
    if (type(seed) is not int or seed not in SEEDS
            or type(train_episodes) is not int or train_episodes!=TRAIN_EPISODES
            or type(transfer_episodes) is not int or transfer_episodes!=TRANSFER_EPISODES
            or type(signed_directions) is not int or signed_directions!=SIGNED_DIRECTIONS
            or type(sigma) not in (int,float) or sigma!=SIGMA):
        raise ValueError("unregistered_transfer_research_trial")


def score_transfer(events):
    transfer=[r for r in events if r["phase"]=="transfer_evaluation"]
    if len(transfer)!=TRANSFER_EPISODES:
        raise ValueError("incomplete_unseen_task_family")
    scores={}
    for name in POLICIES:
        errors=[]
        top=[]
        best=[]
        for row in transfer:
            cells=row["signed_directions"]
            if len(cells)!=SIGNED_DIRECTIONS:
                raise ValueError("selective_model_probing_invalid")
            forecasts=[x["predictions"][name] for x in cells]
            gains=[x["measured_gain"] for x in cells]
            if any(not math.isfinite(v) for v in forecasts+gains):
                raise ValueError("nonfinite_transfer_predictions")
            errors.extend((a-b)**2 for a,b in zip(forecasts,gains,strict=True))
            selected=max(range(len(gains)),key=lambda i:(forecasts[i],-i))
            top.append(gains[selected])
            best.append(max(gains))
        scores[name]={
            "prospective_cases":len(errors),
            "mean_squared_error":statistics.mean(errors),
            "mean_gain_if_top1_tested":statistics.mean(top),
            "mean_oracle_gain":statistics.mean(best),
            "mean_top1_regret":statistics.mean(b-g for b,g in zip(best,top,strict=True)),
        }
    return scores


def run(*,seed,model_path=MODEL_PATH):
    validate(seed=seed)
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3

    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer=AutoTokenizer.from_pretrained(
        str(model_path),trust_remote_code=True,local_files_only=True)
    model,named=_load(torch,seed,str(model_path))
    model.requires_grad_(False)
    model.eval()
    eligible=[(name,p) for name,p in named
              if ".self_attn.v_proj." in name and ".lora_B." in name]
    chosen=sorted(eligible,key=lambda row:
                  int(row[0].split(".layers.")[1].split(".")[0]))[-2:]
    if (len(chosen)!=2
            or sum(p.numel() for _,p in chosen)!=2048
            or any(p.dtype!=torch.float32 for _,p in chosen)
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise RuntimeError("unexpected_trainable_pretrained_parameter")
    params=[p for _,p in chosen]
    originals=[p.detach().clone() for p in params]
    start_hash=_sha(*params)
    basis=_basis(torch,params,seed=seed)
    rng=torch.Generator(device="cpu").manual_seed(987123+seed)

    neural=ForwardOnlyNeuralAttentionHead(seed=seed)
    direction=DirectionAttentionHead()
    activation=ActivationDirectionHead()
    calibrated=CalibratedActivationHead()
    prior=[]
    input_rows=curriculum(seed)
    data_manifest=manifest(seed)
    events=[]
    started=time.perf_counter()
    neural_frozen_sha=None
    neural_frozen_history_size=None
    for ep in range(TRAIN_EPISODES+TRANSFER_EPISODES):
        train=ep<TRAIN_EPISODES
        phase="train" if train else "transfer_evaluation"
        example=(input_rows["train"][ep] if train
                 else input_rows["transfer_evaluation"][ep-TRAIN_EPISODES])
        if train:
            head_training=neural.tune_from_previous(episode=ep)
        elif ep==TRAIN_EPISODES:
            # Last legal head tuning uses only episode 11's train labels.
            head_training=neural.tune_from_previous(episode=ep)
            neural_frozen_sha=neural.parameter_sha256
            neural_frozen_history_size=len(neural.history)
        else:
            head_training=None
        if not train and (neural.parameter_sha256!=neural_frozen_sha
                          or len(neural.history)!=neural_frozen_history_size):
            raise RuntimeError("evaluation_family_changed_neural_head")
        context=prompt_only_activation_context(torch,model,tokenizer,example["prompt"])
        clean=_loss(torch,model,tokenizer,example)
        base_mean=statistics.mean(prior) if prior else 0.
        direction_vectors=[_unit([float(z) for z in torch.randn(
            DIM,generator=rng,dtype=torch.float32)]) for _ in range(4)]
        candidates=[]
        for candidate_idx,v in enumerate(direction_vectors):
            for sign in (+1,-1):
                signed=tuple(sign*x for x in v)
                pd=direction.predict(signed,episode=ep)
                pa=activation.predict(signed,context,episode=ep)
                pc=calibrated.predict(signed,context,episode=ep)
                pn=neural.predict(signed,context,episode=ep)
                candidates.append({
                    "candidate_index":candidate_idx,"sign":sign,
                    "direction":list(signed),
                    "predictions":{
                        "zero":0.0,"historical_mean":base_mean,
                        "direction":pd["predicted_gain"],
                        "activation":pa["predicted_gain"],
                        "calibrated":pc["predicted_gain"],
                        "neural":pn["predicted_gain"],
                    },
                    "neural_prior_attention_keys":pn["historical_keys"],
                    "prior_direction_attention_keys":pd["evidence"],
                })
        # All predictions are frozen before any of this episode's measurements.
        for entry in candidates:
            blocks=_direction(torch,basis,entry["direction"])
            with torch.no_grad():
                for p,orig,d in zip(params,originals,blocks,strict=True):
                    p.copy_(orig+SIGMA*d)
            try:
                perturbed=_loss(torch,model,tokenizer,example)
            finally:
                with torch.no_grad():
                    for p,orig in zip(params,originals,strict=True):
                        p.copy_(orig)
            if _sha(*params)!=start_hash:
                raise RuntimeError("perturbation_not_exactly_restored")
            entry["measured_gain"]=clean-perturbed
            if not math.isfinite(entry["measured_gain"]) or abs(entry["measured_gain"])>100:
                raise RuntimeError("unbounded_forward_only_gain")
        if train:
            # Labels from transfer examples NEVER reach these observers.
            for entry in candidates:
                v=entry["direction"]
                gain=entry["measured_gain"]
                direction.observe(v,gain,episode=ep,split="train",probed=True)
                activation.observe(v,context,gain,episode=ep,split="train",measured=True)
                calibrated.observe(v,context,gain,episode=ep,split="train",measured=True)
                prior.append(gain)
            neural.observe_episode(
                [(x["direction"],context,x["measured_gain"]) for x in candidates],
                episode=ep,split="train",measured=True)
        if not train and (neural.parameter_sha256!=neural_frozen_sha
                          or len(neural.history)!=neural_frozen_history_size):
            raise RuntimeError("transfer_label_leakage")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("unapproved_gradients")
        events.append({
            "episode":ep,"phase":phase,"family":example["family"],
            "example_sha256":hashlib.sha256(json.dumps(
                {k:v for k,v in example.items() if k!="input_key"},
                sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "context":list(context),
            "context_sha256":hashlib.sha256(json.dumps(
                context,separators=(",",":")).encode()).hexdigest(),
            "clean_loss":clean,
            "head_training_before_forecast":head_training,
            "neural_head_sha_after_episode":neural.parameter_sha256,
            "neural_training_memory_after_episode":len(neural.history),
            "signed_directions":candidates,
            "perturbed_forwards":SIGNED_DIRECTIONS,
            "context_only_forwards":1,
            "gradient_backwards":0,
        })
        print(json.dumps({"seed":seed,"episode":ep,"phase":phase,
                          "measured_directions":len(candidates),
                          "head_trainable_195":True}),flush=True)
    if (_sha(*params)!=start_hash or
            any(not torch.equal(p.detach(),o)
                for p,o in zip(params,originals,strict=True))):
        raise RuntimeError("training_model_adapter_modified")
    scores=score_transfer(events)
    return {
        "schema":SCHEMA,"status":"research_disjoint_task_family_forecast_only",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
        "model_run_id":uuid.uuid4().hex,"seed":seed,
        "train_episodes":TRAIN_EPISODES,"transfer_episodes":TRANSFER_EPISODES,
        "training_families":["first_word","last_word"],
        "transfer_family":"uppercase",
        "source_dataset_manifest":data_manifest,
        "initial_adapter_sha256":start_hash,"final_adapter_sha256":_sha(*params),
        "selected_parameter_count":2048,
        "head_parameters":195,
        "head_frozen_before_transfer_sha256":neural_frozen_sha,
        "head_final_sha256":neural.parameter_sha256,
        "head_frozen_before_transfer_memory_size":neural_frozen_history_size,
        "head_model_backward_count":0,
        "model_training_backward_count":0,
        "model_adapter_update_count":0,
        "head_parameter_updates_count":sum(
            x["head_training_before_forecast"]["head_updates_accepted"]
            for x in events if x["head_training_before_forecast"] is not None),
        "source_model_perturbed_forwards":SIGNED_DIRECTIONS*len(events),
        "source_model_context_forwards":len(events),
        "head_only_loss_evaluations":neural.trial_head_forwards,
        "transfer_metrics":scores,
        "events":events,
        "transfer_family_feedback_used_for_training":False,
        "original_selection_partition_opened":False,
        "original_untouched_test_opened":False,
        "new_real_world_task_quality_proven":False,
        "independent_model_execution_attested":False,
        "optimizer_switch_permitted":False,"model_update_permitted":False,
        "weight_promotion_permitted":False,"trial_authorized":False,
        "elapsed_s":max(time.perf_counter()-started,1e-9),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-cross-family-forward-audit",action="store_true",required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run(seed=a.seed,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as handle:
        json.dump(r,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({"seed":a.seed,"status":r["status"],
                      "transfer_neural_mse":r["transfer_metrics"]["neural"]["mean_squared_error"],
                      "transfer_zero_mse":r["transfer_metrics"]["zero"]["mean_squared_error"],
                      "model_adapter_updates":0,
                      "head_frozen_on_transfer":True,"output":str(a.output)}))

if __name__=="__main__":
    main()
