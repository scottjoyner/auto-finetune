"""Preregistered isolated one-step predictive Dust adapter-quality challenge.

Trained 195-parameter attention head from extraction examples chooses ONE
signed LoRA direction before any outcome is measured. Exact same fresh public
train prompt and six independently sampled diagnostic prompts across random,
neural and eight-probe empirical-greedy counterfactual arms. Only a short,
in-memory 2,048-scalar LoRA-B perturbation is evaluated. No persistent model
updates, optimizer switches, checkpoints, production traffic or private data.

The neural head FAILED its independently preregistered zero-gain admission.
Consequently its "selected" update is merely an offline counterfactual,
NEVER a permitted production update. This evaluates a method, not deploys it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from pathlib import Path

from src.lfm3_dust_activation_head import prompt_only_activation_context
from src.lfm3_dust_predictive_probe import _basis, _direction, _loss, _unit
from src.lfm3_dust_text_family_curriculum import WORDS, _one
from src.lfm3_dust_text_family_transfer_review import review as verify_training
from src.lfm3_dust_task_shift_probe import rebuild_training_head, training_source_path
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha
from src.lfm3_dust_matched_gradient_probe import select_b_only

SCHEMA="sensorgraft.lfm3.dust-one-step-in-memory-counterfactual.v1"
NEW_SEEDS=(16,17,18)
OLD_SEEDS=(10,11,12)
PROPOSAL_COUNT=4
SIGNED_DIRECTIONS=8
SIGMA=.05
TRUST_RADIUS=.15
DIAGNOSTIC_SAME_FAMILY=4
DIAGNOSTIC_TRANSFER=2
POLICIES=("no_update","random_one_probe","neural_one_probe","empirical_best_eight_probes")
MAX_REPORT_BYTES=175000


def validate(*,seed,steps=1,sigma=SIGMA):
    if (type(seed) is not int or seed not in NEW_SEEDS
            or type(steps) is not int or steps!=1
            or type(sigma) not in (int,float) or sigma!=SIGMA):
        raise ValueError("unregistered_one_step_sandbox_protocol")


def examples(seed,source):
    validate(seed=seed)
    rng=random.Random(610162+seed)
    seen=set()
    # Reconstruct all 18 historical source IDs, including old transfer data,
    # only to avoid duplicate prompts. We NEVER load transfer gain labels into
    # the head's learning or our direction selection.
    from src.lfm3_dust_text_family_curriculum import curriculum
    historical=curriculum(OLD_SEEDS[NEW_SEEDS.index(seed)])
    seen.update(x["input_key"] for rows in historical.values() for x in rows)
    def fresh(family):
        for _ in range(1000):
            a,b=rng.sample(WORDS,2)
            e=_one(family,a,b)
            if e["input_key"] not in seen:
                seen.add(e["input_key"])
                return e
        raise RuntimeError("no_unique_public_diagnostic_prompt")
    initial=fresh("first_word")
    same=[fresh("first_word" if i%2==0 else "last_word")
          for i in range(DIAGNOSTIC_SAME_FAMILY)]
    transfer=[fresh("uppercase") for _ in range(DIAGNOSTIC_TRANSFER)]
    return {"one_step_train":[initial],
            "diagnostic_same_family":same,
            "diagnostic_transfer":transfer}


def digest(rows):
    return hashlib.sha256(json.dumps(
        [{k:v for k,v in e.items() if k!="input_key"} for e in rows],
        sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def signed_proposals(seed):
    validate(seed=seed)
    gen=random.Random(910017+seed)
    out=[]
    for index in range(PROPOSAL_COUNT):
        v=_unit([gen.gauss(0,1) for _ in range(8)])
        out.extend([{"candidate":index,"sign":1,"direction":list(v)},
                    {"candidate":index,"sign":-1,"direction":[-x for x in v]}])
    return out


def choose_indices(predictions,measured_gains,*,seed):
    if len(predictions)!=SIGNED_DIRECTIONS or len(measured_gains)!=SIGNED_DIRECTIONS:
        raise ValueError("missing_preprobe_or_measured_candidate")
    if any(type(v) not in (float,int) or not math.isfinite(v)
           for v in predictions+measured_gains):
        raise ValueError("nonfinite_gain_score")
    random_index=random.Random(920017+seed).randrange(SIGNED_DIRECTIONS)
    neural_index=max(range(SIGNED_DIRECTIONS),key=lambda i:(predictions[i],-i))
    empirical_index=max(range(SIGNED_DIRECTIONS),key=lambda i:(measured_gains[i],-i))
    return {"no_update":None,
            "random_one_probe":random_index,
            "neural_one_probe":neural_index,
            "empirical_best_eight_probes":empirical_index}


def _diag(torch,model,tokenizer,samples):
    # Two completion-loss diagnostic families. All diagnostic prompts are
    # disjoint from adapter-choice training and from head's 12 train prompts.
    by_family={}
    values=[]
    for name in ("diagnostic_same_family","diagnostic_transfer"):
        losses=[_loss(torch,model,tokenizer,e) for e in samples[name]]
        by_family[name]={"count":len(losses),
                         "mean_completion_loss":statistics.mean(losses)}
        values+=losses
    return {"mean_completion_loss":statistics.mean(values),
            "by_family":by_family,"diagnostic_model_forwards":len(values)}


def run(*,seed,repo_root,model_path=MODEL_PATH):
    validate(seed=seed)
    old=OLD_SEEDS[NEW_SEEDS.index(seed)]
    source_path=training_source_path(repo_root,old)
    if (source_path.is_symlink() or not source_path.is_file()
            or source_path.stat().st_size>MAX_REPORT_BYTES):
        raise RuntimeError("unsafe_training_head_source")
    source_bytes=source_path.read_bytes()
    source=json.loads(source_bytes)
    if verify_training(source).get("status")!="manual_transfer_review_only":
        raise RuntimeError("untrusted_historical_neural_head")
    head=rebuild_training_head(source)
    gate=FrozenTaskShiftGate(source=source)
    if gate.verified_skill is not False:
        # This protocol specifically measures an unqualified head in a
        # counterfactual sandbox, NOT changes admission or production policy.
        raise RuntimeError("unexpected_changed_neural_head_admission")
    initial_head_sha=head.parameter_sha256
    initial_head_memory=len(head.history)
    all_examples=examples(seed,source)
    manifests={name:digest(rows) for name,rows in all_examples.items()}
    if (len({e["input_key"] for rows in all_examples.values() for e in rows})!=7
            or len(set(manifests.values()))!=3):
        raise RuntimeError("nonindependent_model_train_diagnostic_data")
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3
    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer=AutoTokenizer.from_pretrained(
        str(model_path),trust_remote_code=True,local_files_only=True)
    model,named=_load(torch,old,str(model_path))
    selected=select_b_only(named)
    model.requires_grad_(False)
    model.eval()
    params=[p for _,p in selected]
    originals=[p.detach().clone() for p in params]
    initial_adapter=_sha(*params)
    basis=_basis(torch,params,seed=old)
    diagnostics=None
    t0=time.perf_counter()
    actual_forwards=0
    clean_probe=None
    proposals=signed_proposals(seed)
    try:
        # Source LFM3 attention context is prompt-only, before probes.
        step_item=all_examples["one_step_train"][0]
        context=prompt_only_activation_context(
            torch,model,tokenizer,step_item["prompt"])
        initial_context_gate=gate.evaluate(context,0.,candidate_index=0)
        neural_pred=[
            head.predict(x["direction"],context,episode=12)["predicted_gain"]
            for x in proposals
        ]
        predicted_order=sorted(range(SIGNED_DIRECTIONS),
                               key=lambda i:(-neural_pred[i],i))
        baseline=_diag(torch,model,tokenizer,all_examples)
        actual_forwards+=baseline["diagnostic_model_forwards"]
        clean_probe=_loss(torch,model,tokenizer,step_item)
        actual_forwards+=1
        for entry in proposals:
            direction=_direction(torch,basis,entry["direction"])
            with torch.no_grad():
                for p,original,d in zip(params,originals,direction,strict=True):
                    p.copy_(original+SIGMA*d)
            try:
                train_loss=_loss(torch,model,tokenizer,step_item)
                actual_forwards+=1
            finally:
                with torch.no_grad():
                    for p,original in zip(params,originals,strict=True):
                        p.copy_(original)
            if _sha(*params)!=initial_adapter:
                raise RuntimeError("candidate_copy_back_not_exact")
            entry["measured_train_gain"]=clean_probe-train_loss
         # Explicit snapshot alignment; forecast values were computed before
        # *any* candidate gain measurement.
        for i,e in enumerate(proposals):
            e["predicted_train_gain_before_probe"]=neural_pred[i]
        selected_indices=choose_indices(
            neural_pred,[x["measured_train_gain"] for x in proposals],seed=seed)
        arms={}
        for policy in POLICIES:
            idx=selected_indices[policy]
            gain=proposals[idx]["measured_train_gain"] if idx is not None else 0.
            execute=idx is not None and gain>1e-8
            # Safety: validate a single actual training forward before
            # hypothetical in-memory application. No gradients or exports.
            before_ms=time.perf_counter()
            if execute:
                blocks=_direction(torch,basis,proposals[idx]["direction"])
                norm=math.sqrt(sum(float((SIGMA*d).square().sum()) for d in blocks))
                if not 0<norm<=TRUST_RADIUS:
                    raise RuntimeError("trust_radius_exceeded")
                with torch.no_grad():
                    for p,original,d in zip(params,originals,blocks,strict=True):
                        p.copy_(original+SIGMA*d)
            try:
                after=_diag(torch,model,tokenizer,all_examples)
                actual_forwards+=after["diagnostic_model_forwards"]
            finally:
                with torch.no_grad():
                    for p,original in zip(params,originals,strict=True):
                        p.copy_(original)
            if _sha(*params)!=initial_adapter:
                raise RuntimeError("policy_adapter_not_restored")
            arms[policy]={
                "policy":policy,"candidate_preselected_before_train_measurement":(
                    policy=="neural_one_probe"),
                "candidate_index":idx,
                "measured_train_gain_of_candidate":gain,
                "applied_in_memory_for_diagnostic_only":execute,
                "diagnostic_after":after,
                "diagnostic_loss_delta_after_minus_before":(
                    after["mean_completion_loss"]-baseline["mean_completion_loss"]),
                "diagnostic_same_family_delta":(
                    after["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                    baseline["by_family"]["diagnostic_same_family"]["mean_completion_loss"]),
                "diagnostic_transfer_delta":(
                    after["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                    baseline["by_family"]["diagnostic_transfer"]["mean_completion_loss"]),
                "counterfactual_minimum_candidate_probe_forwards":(
                    0 if idx is None else 8 if policy=="empirical_best_eight_probes" else 1),
                "actual_diagnostic_forward_evaluations":after["diagnostic_model_forwards"],
                "elapsed_including_diagnostics_s":time.perf_counter()-before_ms,
                "adapter_restored_sha256":_sha(*params),
                "adapter_exported":False,"production_update_permitted":False,
            }
        if (head.parameter_sha256!=initial_head_sha
                or len(head.history)!=initial_head_memory
                or any(p.grad is not None or p.requires_grad for p in model.parameters())):
            raise RuntimeError("forbidden_neural_head_training_or_model_gradient")
        return {
            "schema":SCHEMA,"status":"isolated_one_step_counterfactual_not_admitted_training",
            "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
            "seed":seed,"source_head_seed":old,"run_id":uuid.uuid4().hex,
            "source_head_training_receipt_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "selected_2048_lora_b_parameters":sum(p.numel() for p in params),
            "initial_adapter_sha256":initial_adapter,
            "final_adapter_sha256":_sha(*params),
            "trained_neural_head_sha256":initial_head_sha,
            "trained_neural_head_parameter_count":195,
            "trained_head_historical_skill_pass":gate.verified_skill,
            "current_train_prompt_in_envelope":initial_context_gate["in_training_context_envelope"],
            "head_novelty_guard_decision":initial_context_gate,
            "new_train_prompt_lfm3_context":list(context),
            "train_and_diagnostic_sha256":manifests,
            "diagnostic_example_count":6,"single_train_prompt_count":1,
            "model_probe_directions_measured":8,
            "model_candidate_perturbation_forwards":8,
            "model_clean_probe_forwards":1,
            "model_diagnostic_forwards":actual_forwards-9,
            "total_model_loss_forwards":actual_forwards,
            "model_prompt_context_forwards":1,
            "head_loss_evaluations_in_this_run":0,
            "model_backward_calls":0,
            "persistent_adapter_updates":0,
            "models_or_adapters_saved":0,
            "neural_head_learned_from_new_diagnostic_labels":False,
            "selection_test_data_opened":False,
            "independent_execution_attested":False,
            "predicted_neural_candidate_order":predicted_order,
            "measured_candidate_outcomes":proposals,
            "baseline_diagnostic":baseline,
            "policy_results":arms,
            "counterfactuals_are_not_equal_walltime_or_energy":True,
            "this_report_is_not_a_three_way_adamw_muon_budget_match":True,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
            "production_execution_permitted":False,
            "elapsed_s":max(time.perf_counter()-t0,1e-9),
        }
    finally:
        with torch.no_grad():
            for p,original in zip(params,originals,strict=True):
                p.copy_(original)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-isolated-unqualified-head-counterfactual",action="store_true",required=True)
    p.add_argument("--seed",type=int,choices=NEW_SEEDS,required=True)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=run(seed=a.seed,repo_root=a.root,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({
        "seed":a.seed,"status":out["status"],
        "random_delta":out["policy_results"]["random_one_probe"]["diagnostic_loss_delta_after_minus_before"],
        "neural_delta":out["policy_results"]["neural_one_probe"]["diagnostic_loss_delta_after_minus_before"],
        "greedy_delta":out["policy_results"]["empirical_best_eight_probes"]["diagnostic_loss_delta_after_minus_before"],
        "persistent_adapter_updates":0,"output":str(a.output),
    },allow_nan=False))


if __name__=="__main__":
    main()
