"""Fresh task-shift probe on real LFM3 attention, NO perturbation forwards.

Preregistered alphabetization-of-two-words instructions are new relative to
first/last extraction and uppercase transfer. Three seed-specific cohorts,
six public prompts each. A head trained entirely on original extraction
training episodes is FROZEN; its raw forecasts are optionally scored by a
TRAIN-ONLY attention-context novelty and forecast-skill guard.

No new model updates, gain labels, probing, optimizer steps, heldout scoring,
private traces or production requests. We assess detection/abstention only,
NOT forecast MSE or downstream model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path

from src.lfm3_dust_activation_head import prompt_only_activation_context
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_trainable_attention import ForwardOnlyNeuralAttentionHead
from src.lfm3_dust_text_family_curriculum import WORDS
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA="sensorgraft.lfm3.dust-new-family-novelty-shadow.v1"
NEW_SEEDS=(13,14,15)
OLD_SEEDS=(10,11,12)
EPISODES=6
CANDIDATES=8


def curriculum(seed):
    if type(seed) is not int or seed not in NEW_SEEDS:
        raise ValueError("unregistered_novelty_trial_seed")
    rng=random.Random(931107+seed)
    seen=set()
    samples=[]
    for i in range(EPISODES):
        for _ in range(500):
            a,b=rng.sample(WORDS,2)
            key="sort_two_words|"+a+"|"+b
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "family":"sort_two_words",
                "input_key":key,
                "prompt":("Follow the instruction exactly. Sort these two lowercase words "
                          f"alphabetically and return only the two words separated by a space. Words: {a} {b}\nAnswer:"),
            })
            break
        else:
            raise RuntimeError("failed_to_sample_distinct_public_task")
    return samples


def task_hash(item):
    return hashlib.sha256(json.dumps({k:v for k,v in item.items() if k!="input_key"},
                                     sort_keys=True,separators=(",",":")).encode()).hexdigest()


def training_source_path(repo,old_seed):
    return repo/("research/evidence/lfm3-dust-text-transfer-20261029/"
                 f"seed{old_seed}-extraction-to-uppercase-18episodes.json")


def rebuild_training_head(source):
    seed=source["seed"]
    head=ForwardOnlyNeuralAttentionHead(seed=seed)
    for ep in range(12):
        event=head.tune_from_previous(episode=ep)
        row=source["events"][ep]
        if event!=row["head_training_before_forecast"]:
            raise RuntimeError("unreproducible_pretrained_head")
        head.observe_episode(
            [(c["direction"],row["context"],c["measured_gain"])
             for c in row["signed_directions"]],
            episode=ep,split="train",measured=True)
        if head.parameter_sha256!=row["neural_head_sha_after_episode"]:
            raise RuntimeError("head_recovery_hash_mismatch")
    head.tune_from_previous(episode=12)
    if head.parameter_sha256!=source["head_frozen_before_transfer_sha256"]:
        raise RuntimeError("unverified_neural_head_freeze")
    return head


def directions(seed,ep):
    rng=random.Random(706117+100*seed+ep)
    signed=[]
    for i in range(CANDIDATES//2):
        vec=[rng.gauss(0,1) for _ in range(8)]
        norm=math.sqrt(sum(x*x for x in vec))
        vec=[x/norm for x in vec]
        signed.extend([(i,1,vec),(i,-1,[-x for x in vec])])
    return signed


def run(*,seed,repo_root,model_path=MODEL_PATH):
    if type(seed) is not int or seed not in NEW_SEEDS:
        raise ValueError("nonregistered_new_family_seed")
    old=OLD_SEEDS[NEW_SEEDS.index(seed)]
    source_file=training_source_path(repo_root,old)
    if (source_file.is_symlink() or not source_file.is_file()
            or source_file.stat().st_size>175000):
        raise RuntimeError("untrusted_training_source")
    raw=source_file.read_bytes()
    source=json.loads(raw)
    from src.lfm3_dust_text_family_transfer_review import review
    if review(source).get("status")!="manual_transfer_review_only":
        raise RuntimeError("unverified_prior_training_head")
    head=rebuild_training_head(source)
    gate=FrozenTaskShiftGate(source=source)

    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3
    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),
                       trust_remote_code=True,local_files_only=True)
    model,named=_load(torch,old,str(model_path))
    model.requires_grad_(False)
    model.eval()
    selected=sorted([(name,p) for name,p in named
          if ".self_attn.v_proj." in name and ".lora_B." in name],
          key=lambda x:int(x[0].split(".layers.")[1].split(".")[0]))[-2:]
    if (len(selected)!=2 or sum(p.numel() for _,p in selected)!=2048
            or any(p.grad is not None or p.requires_grad for p in model.parameters())):
        raise RuntimeError("unapproved_model_training_or_wrong_adapter")
    initial=_sha(*(p for _,p in selected))
    head_initial=head.parameter_sha256
    history_initial=len(head.history)
    samples=curriculum(seed)
    result=[]
    t0=time.perf_counter()
    for ep,item in enumerate(samples):
        ctx=prompt_only_activation_context(torch,model,tokenizer,item["prompt"])
        preds=[]
        for candidate,sign,v in directions(seed,ep):
            p=head.predict(v,ctx,episode=12+ep)
            decision=gate.evaluate(ctx,p["predicted_gain"],candidate_index=len(preds))
            preds.append({"candidate":candidate,"sign":sign,
                          "direction":v,"raw_neural_prediction":p["predicted_gain"],
                          "head_keys":p["historical_keys"],"guard":decision,
                          "actual_perturbation_forward_performed":False,
                          "measured_gain":None})
        if (head.parameter_sha256!=head_initial
                or len(head.history)!=history_initial
                or _sha(*(p for _,p in selected))!=initial):
            raise RuntimeError("model_or_head_changed_during_novelty_run")
        result.append({
            "episode":ep,"family":"sort_two_words",
            "prompt_sha256":task_hash(item),"context":list(ctx),
            "raw_head_predicted_directions":preds,
            "labeled_gain_observations":0,
            "actual_model_context_forwards":1,
            "actual_perturbation_forwards":0,
            "head_parameter_updates":0,
            "model_backward_calls":0,
        })
        print(json.dumps({"seed":seed,"episode":ep,
                          "ood":not preds[0]["guard"]["in_training_context_envelope"],
                          "abstained":all(p["guard"]["abstained"] for p in preds)}),flush=True)
    return {
        "schema":SCHEMA,"status":"research_unseen_family_novelty_shadow_only",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
        "source_training_seed":old,"new_task_seed":seed,"run_id":uuid.uuid4().hex,
        "head_parameters":195,"head_frozen_sha256":head.parameter_sha256,
        "head_history_size":len(head.history),
        "source_training_receipt_sha256":hashlib.sha256(raw).hexdigest(),
        "source_model_selected_adapter_parameters":2048,
        "adapter_initial_sha256":initial,"adapter_final_sha256":initial,
        "frozen_train_only_guard":gate.summary(),
        "novel_family":"sort_two_words","new_family_episodes":EPISODES,
        "source_model_prompt_only_forwards":EPISODES,
        "source_model_perturbation_forwards":0,
        "source_model_backward_calls":0,
        "retained_model_adapter_updates":0,
        "additional_head_parameter_updates":0,
        "candidate_gain_labels_collected":0,
        "original_selection_partition_opened":False,
        "original_test_partition_opened":False,
        "model_quality_improvement_proven":False,
        "forecast_error_improvement_proven":False,
        "new_family_novelty_detection_only":True,
        "source_runtime_independently_attested":False,
        "optimizer_switch_permitted":False,
        "model_training_permitted":False,
        "weight_promotion_permitted":False,
        "episodes":result,"elapsed_s":max(time.perf_counter()-t0,1e-9),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-novelty-context-only",action="store_true",required=True)
    p.add_argument("--seed",type=int,choices=NEW_SEEDS,required=True)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    output=run(seed=a.seed,repo_root=a.root,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({"seed":a.seed,"status":output["status"],
                      "prompt_only_model_forwards":EPISODES,
                      "model_weight_updates":0,"output":str(a.output)}))


if __name__=="__main__":
    main()
