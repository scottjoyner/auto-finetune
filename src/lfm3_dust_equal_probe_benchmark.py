"""Actual equal-probe-budget LFM3 predictive Dust versus random control.

Registered seeds 25/26/27. Each research policy gets one signed model
direction, one measured training perturbation and six diagnostic loss
forwards. Neural direction is chosen BEFORE any new outcome; a random
direction is chosen from the same deterministic candidate set. Both
have precisely 1 model probe, no hidden greedy eight-probe oracle.

The 195-weight pretrained external attention predictor remains frozen, as
does the official BF16 LFM3-1.7B backbone. All 2,048 LoRA-B perturbations are
temporary and exactly restored. Head-attention context forward and fitting
from historical training receipts are counted separately from model-loss
forwards and policy timing. The earlier admission remains HOLD and none
of these isolated counterfactual steps is a production training update.
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

from src.lfm3_dust_predictive_probe import _basis,_direction,_loss,_unit
from src.lfm3_dust_activation_head import prompt_only_activation_context
from src.lfm3_dust_sandbox_update_quality import _diag,digest
from src.lfm3_dust_text_family_curriculum import WORDS,_one,curriculum
from src.lfm3_dust_text_family_transfer_review import review as verify_source
from src.lfm3_dust_task_shift_probe import rebuild_training_head,training_source_path
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_matched_gradient_probe import select_b_only
from src.lfm3_real_lora_probe import BASE_SHA256,MODEL_PATH,_load,_preflight,_sha

SCHEMA="sensorgraft.lfm3.dust-equal-one-probe-budget.v1"
SEEDS=(25,26,27)
HEAD_SEEDS=(10,11,12)
SIGMA=.05
MAX_RADIUS=.15
SIGNED_CANDIDATES=8
POLICIES=("random_one_probe","neural_one_probe")
MAX_ORIGINAL_RECEIPT=175000


def validate(*,seed,probes_per_policy=1,sigma=SIGMA):
    if (type(seed) is not int or seed not in SEEDS
            or type(probes_per_policy) is not int or probes_per_policy!=1
            or type(sigma) not in (int,float) or sigma!=SIGMA):
        raise ValueError("unregistered_equal_probe_research_protocol")


def task_examples(seed):
    validate(seed=seed)
    head_seed=HEAD_SEEDS[SEEDS.index(seed)]
    old=curriculum(head_seed)
    taken={e["input_key"] for rows in old.values() for e in rows}
    rng=random.Random(2026103025+seed)
    def sample(family):
        for _ in range(1000):
            a,b=rng.sample(WORDS,2)
            item=_one(family,a,b)
            if item["input_key"] not in taken:
                taken.add(item["input_key"])
                return item
        raise RuntimeError("unable_to_sample_unique_public_example")
    return {
        "training":[sample("first_word")],
        "diagnostic_same_family":[sample("first_word"),sample("last_word"),
                                  sample("first_word"),sample("last_word")],
        "diagnostic_transfer":[sample("uppercase"),sample("uppercase")],
    }


def candidate_set(seed):
    validate(seed=seed)
    rng=random.Random(1103225+seed)
    rows=[]
    for index in range(4):
        vec=list(_unit([rng.gauss(0,1) for _ in range(8)]))
        rows.extend([
            {"pair":index,"sign":1,"direction":vec},
            {"pair":index,"sign":-1,"direction":[-x for x in vec]},
        ])
    return rows


def choose(seed,predictions):
    validate(seed=seed)
    if (not isinstance(predictions,(tuple,list)) or len(predictions)!=8
            or any(type(x) not in (int,float) or not math.isfinite(x)
                   for x in predictions)):
        raise ValueError("invalid_preactuation_forecasts")
    return {
        "random_one_probe":random.Random(133025+seed).randrange(8),
        "neural_one_probe":max(range(8),key=lambda i:(predictions[i],-i)),
    }


def run(*,seed,repo_root,model_path=MODEL_PATH):
    validate(seed=seed)
    head_seed=HEAD_SEEDS[SEEDS.index(seed)]
    path=training_source_path(Path(repo_root),head_seed)
    if (path.is_symlink() or not path.is_file() or
            not 100<path.stat().st_size<=MAX_ORIGINAL_RECEIPT):
        raise RuntimeError("invalid_source_training_receipt")
    source_bytes=path.read_bytes()
    source=json.loads(source_bytes)
    if verify_source(source).get("status")!="manual_transfer_review_only":
        raise RuntimeError("unreviewed_original_head_trajectory")
    started_head=time.perf_counter()
    head=rebuild_training_head(source)
    gate=FrozenTaskShiftGate(source=source)
    head_reconstruction_s=time.perf_counter()-started_head
    if gate.verified_skill:
        raise RuntimeError("unanticipated_admitted_neural_head")
    frozen_head_hash=head.parameter_sha256
    frozen_head_memory=len(head.history)
    examples=task_examples(seed)
    manifests={name:digest(items) for name,items in examples.items()}
    if len({e["input_key"] for rows in examples.values() for e in rows})!=7:
        raise RuntimeError("overlapping_training_and_diagnostics")
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3
    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer=AutoTokenizer.from_pretrained(
        str(model_path),trust_remote_code=True,local_files_only=True)
    model,named=_load(torch,head_seed,str(model_path))
    selected=select_b_only(named)
    model.requires_grad_(False)
    model.eval()
    params=[p for _,p in selected]
    originals=[p.detach().clone() for p in params]
    initial_sha=_sha(*params)
    basis=_basis(torch,params,seed=head_seed)
    if sum(p.numel() for p in params)!=2048:
        raise RuntimeError("mismatched_adapter_parameter_scope")

    def apply(candidate,callback):
        v=candidate["direction"]
        blocks=_direction(torch,basis,v)
        norm=math.sqrt(sum(float((SIGMA*d).square().sum()) for d in blocks))
        if not 0<norm<=MAX_RADIUS:
            raise RuntimeError("unsafe_adapter_step_radius")
        with torch.no_grad():
            for p,base,delta in zip(params,originals,blocks,strict=True):
                p.copy_(base+SIGMA*delta)
        try:
            return callback()
        finally:
            with torch.no_grad():
                for p,base in zip(params,originals,strict=True):
                    p.copy_(base)
            if _sha(*params)!=initial_sha:
                raise RuntimeError("temporary_adapter_restore_failed")
    start=time.perf_counter()
    try:
        train=examples["training"][0]
        context_start=time.perf_counter()
        context=prompt_only_activation_context(torch,model,tokenizer,
                                                train["prompt"])
        context_elapsed=time.perf_counter()-context_start
        gate_decision=gate.evaluate(context,0.,candidate_index=0)
        directions=candidate_set(seed)
        forecast_start=time.perf_counter()
        predicted=[head.predict(c["direction"],context,episode=12)["predicted_gain"]
                   for c in directions]
        head_inference_elapsed=time.perf_counter()-forecast_start
        chosen=choose(seed,predicted)
        clean_train=_loss(torch,model,tokenizer,train)
        baseline=_diag(torch,model,tokenizer,examples)
        arms={}
        # Both policies pay exactly one training-probe plus six diagnostics,
        # including when a negative training gain leads to abstention.
        for name in POLICIES:
            i=chosen[name]
            pre_sha=_sha(*params)
            t0=time.perf_counter()
            train_loss=apply(directions[i],lambda: _loss(
                torch,model,tokenizer,train))
            measured_gain=clean_train-train_loss
            accepted=measured_gain>1e-8
            score=(apply(directions[i],lambda:_diag(torch,model,tokenizer,examples))
                   if accepted else _diag(torch,model,tokenizer,examples))
            policy_elapsed=time.perf_counter()-t0
            if _sha(*params)!=pre_sha or pre_sha!=initial_sha:
                raise RuntimeError("adapter_tainted_between_policies")
            arms[name]={
                "policy":name,
                "selected_signed_candidate":i,
                "selected_direction":directions[i],
                "selected_prediction_before_any_probe":predicted[i],
                "training_clean_completion_loss":clean_train,
                "training_perturbed_completion_loss":train_loss,
                "measured_training_gain":measured_gain,
                "positive_gain_acceptance":accepted,
                "temporary_adapter_step_diagnosed":accepted,
                "diagnostic_after":score,
                "diagnostic_completion_loss_delta":(
                    score["mean_completion_loss"]-baseline["mean_completion_loss"]),
                "diagnostic_same_family_loss_delta":(
                    score["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                    baseline["by_family"]["diagnostic_same_family"]["mean_completion_loss"]),
                "diagnostic_transfer_loss_delta":(
                    score["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                    baseline["by_family"]["diagnostic_transfer"]["mean_completion_loss"]),
                "model_probe_forwards":1,
                "model_diagnostic_forwards":6,
                "model_backwards":0,
                "policy_probe_and_diagnostic_wall_s":policy_elapsed,
                "adapter_restored_sha256":_sha(*params),
                "retained_model_update":False,
                "checkpoint_saved":False,
            }
        if (head.parameter_sha256!=frozen_head_hash
                or len(head.history)!=frozen_head_memory
                or _sha(*params)!=initial_sha
                or any(p.requires_grad or p.grad is not None
                       for p in model.parameters())):
            raise RuntimeError("training_forbidden_or_head_mutated")
        return {
            "schema":SCHEMA,
            "status":"read_only_equal_one_probe_model_counterfactual",
            "seed":seed,"source_head_seed":head_seed,"run_id":uuid.uuid4().hex,
            "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
            "original_head_receipt_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "head_parameters":195,"head_frozen_sha256":frozen_head_hash,
            "head_training_memory_size":frozen_head_memory,
            "head_historical_gain_skill_qualified":False,
            "model_lora_b_scope":2048,
            "initial_adapter_sha256":initial_sha,
            "final_adapter_sha256":_sha(*params),
            "task_sha256_manifests":manifests,
            "training_prompt_count":1,"diagnostic_prompt_count":6,
            "diagnostic_partitions":["diagnostic_same_family","diagnostic_transfer"],
            "preprobe_context":list(context),
            "preprobe_context_gate":gate_decision,
            "candidate_directions":directions,
            "all_preprobe_neural_gain_forecasts":predicted,
            "selected_indices_from_prehistory":chosen,
            "clean_training_loss":clean_train,
            "baseline_diagnostic":baseline,
            "policy_results":arms,
            "measured_training_candidate_forwards":2,
            "measured_training_baseline_forwards":1,
            "measured_diagnostic_model_forwards":18,
            "total_loss_model_forwards":21,
            "neural_extra_prompt_context_forwards":1,
            "random_extra_prompt_context_forwards":0,
            "neural_head_reconstruction_elapsed_s":head_reconstruction_s,
            "neural_prompt_context_elapsed_s":context_elapsed,
            "neural_head_forecast_elapsed_s":head_inference_elapsed,
            "total_experiment_inference_elapsed_s":time.perf_counter()-start,
            "actual_budget_per_policy_model_probe_forwards":1,
            "actual_budget_per_policy_diagnostic_forwards":6,
            "policy_model_probe_and_diagnostic_budgets_equal":True,
            "model_load_and_shared_baseline_cost_not_per_policy":True,
            "policy_execution_order":list(POLICIES),
            "observed_walltime_not_energy_normalized":True,
            "neural_head_receipt_custody_independently_attested":False,
            "new_model_trial_custody_independently_attested":False,
            "original_locked_selection_test_opened":False,
            "head_fine_tuned_on_new_trial_outcomes":False,
            "model_backward_count":0,"persistent_lora_update_count":0,
            "model_checkpoint_saved":False,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
        }
    finally:
        with torch.no_grad():
            for p,base in zip(params,originals,strict=True):
                p.copy_(base)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-frozen-head-equal-probe-counterfactual",
                   action="store_true",required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=run(seed=a.seed,repo_root=a.root,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps({
        "seed":a.seed,"status":result["status"],
        "random_diagnostic_loss_delta":result["policy_results"]["random_one_probe"]["diagnostic_completion_loss_delta"],
        "neural_diagnostic_loss_delta":result["policy_results"]["neural_one_probe"]["diagnostic_completion_loss_delta"],
        "measured_loss_forwards":result["total_loss_model_forwards"],
        "model_weights_retained":0,"output":str(a.output),
    },allow_nan=False))


if __name__=="__main__":
    main()
