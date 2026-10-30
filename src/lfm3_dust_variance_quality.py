"""Preregistered gain-variance-aware Dust LoRA-B acceptance research.

A genuinely trained external 195-weight LFM3 attention head ranks eight
signed LoRA-B perturbations BEFORE new outcomes. Measure the first training
prompt for all eight; compare the one-prompt accept-if-positive rule against
a three-prompt all-positive rule and a variance-aware rule requiring
all three gains positive AND mean gain minus one sample standard deviation > 0. Same random, neural and eight-probe greedy
candidates for each rule; independent six-prompt diagnostics assess every
candidate that passes the one-prompt rule, even when multi-prompt vetoes it.

Strict isolated CPU, public procedural prompts, frozen official LFM3-1.7B.
Every perturbation is ephemeral and exactly restored. No gradients, model
checkpoint writes, auto optimizer switching, selection/final-test scoring,
new head training, private data or production changes. The neural controller
remains UNQUALIFIED from earlier evaluation and cannot actuate live training.
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

from src.lfm3_dust_predictive_probe import _basis, _direction, _loss
from src.lfm3_dust_activation_head import prompt_only_activation_context
from src.lfm3_dust_sandbox_update_quality import (
    signed_proposals as _previous_proposals, _diag, digest,
)
from src.lfm3_dust_text_family_curriculum import WORDS, _one, curriculum
from src.lfm3_dust_text_family_transfer_review import review as verify_head_source
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_task_shift_probe import (
    rebuild_training_head, training_source_path,
)
from src.lfm3_dust_matched_gradient_probe import select_b_only
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA="sensorgraft.lfm3.dust-variance-aware-one-step-quality.v1"
NEW_SEEDS=(22,23,24)
HEAD_SEEDS=(10,11,12)
TRAINING_PROMPTS=3
DIAGNOSTIC_PROMPTS=6
CANDIDATES=8
SIGMA=.05
TRUST=.15
POLICIES=(
    "no_update",
    "random_single","random_multi","random_variance",
    "neural_single","neural_multi","neural_variance",
    "greedy_single","greedy_multi","greedy_variance",
)


def validate(*,seed,training_prompts=TRAINING_PROMPTS,sigma=SIGMA):
    if (type(seed) is not int or seed not in NEW_SEEDS
            or type(training_prompts) is not int
            or training_prompts!=TRAINING_PROMPTS
            or type(sigma) not in (int,float) or sigma!=SIGMA):
        raise ValueError("unregistered_multi_prompt_research_protocol")


def samples_for(seed):
    validate(seed=seed)
    head_seed=HEAD_SEEDS[NEW_SEEDS.index(seed)]
    old=curriculum(head_seed)
    seen={row["input_key"] for split in old.values() for row in split}
    rng=random.Random(10261031+seed)
    def fresh(family):
        for _ in range(1000):
            a,b=rng.sample(WORDS,2)
            row=_one(family,a,b)
            if row["input_key"] not in seen:
                seen.add(row["input_key"])
                return row
        raise RuntimeError("unique_task_generation_failed")
    return {
        "training":[fresh("first_word"),fresh("last_word"),fresh("first_word")],
        "diagnostic_same_family":[fresh("first_word"),fresh("last_word"),
                                  fresh("first_word"),fresh("last_word")],
        "diagnostic_transfer":[fresh("uppercase") for _ in range(2)],
    }


def proposals_for(seed):
    validate(seed=seed)
    # Existing source pre-registered candidate generator was seed 16..18.
    # New independent seed 19..21 generates distinct deterministic directions.
    from src.lfm3_dust_predictive_probe import _unit
    gen=random.Random(910018+seed)
    result=[]
    for i in range(4):
        v=list(_unit([gen.gauss(0,1) for _ in range(8)]))
        result += [{"candidate":i,"sign":1,"direction":v},
                   {"candidate":i,"sign":-1,"direction":[-x for x in v]}]
    return result


def choices(predictions,first_gains,*,seed):
    if (len(predictions)!=CANDIDATES or len(first_gains)!=CANDIDATES
            or any(type(x) not in (int,float) or not math.isfinite(x)
                   for x in predictions+first_gains)):
        raise ValueError("unverified_preprompt_forecast_or_gain")
    return {
        "random":random.Random(920018+seed).randrange(CANDIDATES),
        "neural":max(range(CANDIDATES),key=lambda i:(predictions[i],-i)),
        "greedy":max(range(CANDIDATES),key=lambda i:(first_gains[i],-i)),
    }


def verdicts(primary,confirmation):
    if type(primary) not in (int,float) or not math.isfinite(primary):
        raise ValueError("invalid_first_training_gain")
    one=primary>1e-8
    if not one:
        if confirmation is not None:
            raise ValueError("must_not_measure_unneeded_confirmations")
        return {"single":False,"multi":False,
                "reason":"first_training_gain_nonpositive"}
    if (not isinstance(confirmation,(list,tuple))
            or len(confirmation)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v)
                   or abs(v)>100 for v in confirmation)):
        raise ValueError("unverified_second_and_third_training_gain")
    multi=all(v>1e-8 for v in confirmation)
    return {"single":True,"multi":multi,
            "reason":"three_train_prompts_positive" if multi
                     else "confirmation_training_gain_nonpositive"}


def variance_verdict(primary,confirmation):
    """Preregistered heuristic proxy, NOT a statistical confidence interval.

    Accept only if each of three train-side gains is positive AND
    sample mean minus one sample standard deviation remains positive.
    Uses no diagnostic labels. This can still accept harmful updates.
    """
    d=verdicts(primary,confirmation)
    if not d["multi"]:
        return False
    vals=[primary,*confirmation]
    return statistics.mean(vals)-statistics.stdev(vals)>1e-8


def run(*,seed,repo_root,model_path=MODEL_PATH):
    validate(seed=seed)
    old=HEAD_SEEDS[NEW_SEEDS.index(seed)]
    path=training_source_path(repo_root,old)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>175000:
        raise RuntimeError("unsafe_historical_training_source")
    raw=path.read_bytes()
    source=json.loads(raw)
    if verify_head_source(source).get("status")!="manual_transfer_review_only":
        raise RuntimeError("head_training_evidence_unqualified")
    head=rebuild_training_head(source)
    gate=FrozenTaskShiftGate(source=source)
    if gate.verified_skill:
        raise RuntimeError("unregistered_admitted_neural_head")
    frozen_head_sha=head.parameter_sha256
    frozen_memory=len(head.history)
    samples=samples_for(seed)
    manifests={k:digest(v) for k,v in samples.items()}
    if len({e["input_key"] for rows in samples.values() for e in rows})!=9:
        raise RuntimeError("train_diagnostic_prompt_overlap")
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
    initial_sha=_sha(*params)
    basis=_basis(torch,params,seed=old)
    time_start=time.perf_counter()
    loss_forwards=0
    context_forwards=0

    def candidate_loss(example,index):
        nonlocal loss_forwards
        v=proposals[index]["direction"]
        blocks=_direction(torch,basis,v)
        norm=math.sqrt(sum(float((SIGMA*d).square().sum()) for d in blocks))
        if not 0<norm<=TRUST:
            raise RuntimeError("unbounded_proposed_adapter_step")
        with torch.no_grad():
            for p,orig,d in zip(params,originals,blocks,strict=True):
                p.copy_(orig+SIGMA*d)
        try:
            loss=_loss(torch,model,tokenizer,example)
            loss_forwards+=1
            return loss
        finally:
            with torch.no_grad():
                for p,orig in zip(params,originals,strict=True):
                    p.copy_(orig)
            if _sha(*params)!=initial_sha:
                raise RuntimeError("temporary_adapter_not_exactly_restored")

    def diagnostic_for(index):
        nonlocal loss_forwards
        blocks=_direction(torch,basis,proposals[index]["direction"])
        norm=math.sqrt(sum(float((SIGMA*d).square().sum()) for d in blocks))
        if not 0<norm<=TRUST:
            raise RuntimeError("unbounded_diagnostic_adapter_step")
        with torch.no_grad():
            for p,orig,d in zip(params,originals,blocks,strict=True):
                p.copy_(orig+SIGMA*d)
        try:
            score=_diag(torch,model,tokenizer,samples)
            loss_forwards+=score["diagnostic_model_forwards"]
            return score
        finally:
            with torch.no_grad():
                for p,orig in zip(params,originals,strict=True):
                    p.copy_(orig)
            if _sha(*params)!=initial_sha:
                raise RuntimeError("diagnostic_adapter_not_exactly_restored")

    try:
        first=samples["training"][0]
        ctx=prompt_only_activation_context(torch,model,tokenizer,first["prompt"])
        context_forwards+=1
        shift_guard=gate.evaluate(ctx,0.,candidate_index=0)
        proposals=proposals_for(seed)
        forecasts=[head.predict(v["direction"],ctx,episode=12)["predicted_gain"]
                   for v in proposals]
        # All 8 prospective forecasts frozen before measuring ANY new gain.
        before=_diag(torch,model,tokenizer,samples)
        loss_forwards+=before["diagnostic_model_forwards"]
        clean=[_loss(torch,model,tokenizer,ex) for ex in samples["training"]]
        loss_forwards+=len(clean)
        for i,entry in enumerate(proposals):
            entry["predicted_gain_before_any_measurement"]=forecasts[i]
            entry["measured_first_train_gain"]=clean[0]-candidate_loss(first,i)
        selected=choices(forecasts,[x["measured_first_train_gain"]
                                    for x in proposals],seed=seed)
        confirm={}
        quality={}
        for idx in set(selected.values()):
            first_gain=proposals[idx]["measured_first_train_gain"]
            if first_gain>1e-8:
                # Second and third training prompts serve solely as a
                # candidate acceptance check, never for predictor learning.
                confirmations=[clean[k]-candidate_loss(samples["training"][k],idx)
                               for k in (1,2)]
                confirm[str(idx)]=confirmations
                # Independently score the positive first-prompt proposal
                # even if the new guard rejects it. This reveals false-positive
                # single-prompt updates without leaking diagnostic labels
                # into the acceptance decision.
                quality[str(idx)]=diagnostic_for(idx)
            else:
                confirm[str(idx)]=None
        arms={"no_update":{
            "candidate_index":None,"training_gain_first":0.,
            "training_confirmation_gains":None,
            "single_prompt_would_apply":False,
            "multi_prompt_would_apply":False,
            "diagnostic_after":before,
            "diagnostic_mean_loss_delta":0.,
            "guard_reason":"no_update",
            "retained_adapter_updates":0,
        }}
        for name,idx in selected.items():
            decision=verdicts(proposals[idx]["measured_first_train_gain"],
                              confirm[str(idx)])
            decision["variance"]=variance_verdict(proposals[idx]["measured_first_train_gain"],confirm[str(idx)])
            for mode in ("single","multi","variance"):
                apply=decision[mode]
                after=quality[str(idx)] if apply else before
                arms[f"{name}_{mode}"]={
                    "candidate_index":idx,
                    "training_gain_first":proposals[idx]["measured_first_train_gain"],
                    "training_confirmation_gains":confirm[str(idx)],
                    "single_prompt_would_apply":decision["single"],
                    "multi_prompt_would_apply":decision["multi"],
                    "would_apply_under_own_rule":apply,
                    "diagnostic_after":after,
                    "diagnostic_mean_loss_delta":(
                        after["mean_completion_loss"]-before["mean_completion_loss"]),
                    "guard_reason":decision["reason"],
                    "retained_adapter_updates":0,
                }
        if (head.parameter_sha256!=frozen_head_sha
                or len(head.history)!=frozen_memory
                or any(p.grad is not None or p.requires_grad
                       for p in model.parameters())
                or _sha(*params)!=initial_sha):
            raise RuntimeError("model_or_head_training_not_frozen")
        return {
            "schema":SCHEMA,
            "status":"offline_unqualified_head_variance_counterfactual",
            "seed":seed,"trained_head_seed":old,"run_id":uuid.uuid4().hex,
            "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
            "source_training_receipt_sha256":hashlib.sha256(raw).hexdigest(),
            "source_training_head_sha256":frozen_head_sha,
            "source_head_memory_entries":frozen_memory,
            "trained_attention_parameters":195,
            "selected_model_adapter_parameters":sum(p.numel() for p in params),
            "initial_adapter_sha256":initial_sha,
            "final_adapter_sha256":_sha(*params),
            "training_prompts":TRAINING_PROMPTS,"diagnostic_prompts":DIAGNOSTIC_PROMPTS,
            "variance_gate_definition":"all_three_positive_and_sample_mean_minus_one_sample_std_positive",
            "train_diagnostic_manifest":manifests,
            "new_train_prompt_attention_context":list(ctx),
            "historical_head_accepted_by_gain_skill":False,
            "train_only_shift_guard_decision":shift_guard,
            "model_gradients_and_backprop_calls":0,
            "head_training_updates_this_run":0,
            "persistent_model_adapter_updates":0,
            "models_or_adapters_saved":0,
            "selection_test_partition_opened":False,
            "source_runtime_independently_attested":False,
            "diagnostic_labels_used_for_head_or_gate":False,
            "candidate_direction_count":CANDIDATES,
            "candidate_first_prompt_forwards":CANDIDATES,
            "training_clean_model_forwards":TRAINING_PROMPTS,
            "diagnostic_baseline_model_forwards":DIAGNOSTIC_PROMPTS,
            "training_confirmation_model_forwards":2*sum(v is not None
                                                          for v in confirm.values()),
            "diagnostic_candidate_model_forwards":DIAGNOSTIC_PROMPTS*len(quality),
            "total_model_loss_forwards":loss_forwards,
            "total_model_prompt_context_forwards":context_forwards,
            "proposals":proposals,
            "selected_candidate_index_by_policy":selected,
            "candidate_confirmation_gains":confirm,
            "positive_first_gain_candidate_diagnostics":quality,
            "baseline_diagnostic":before,
            "results":arms,
            "comparison_is_not_compute_matched_to_adamw_muon":True,
            "single_and_multi_comparison_reuses_identical_candidate_outcomes":True,
            "variance_acceptance_is_unverified_heuristic":True,
            "production_optimizer_switch_permitted":False,
            "model_update_permitted":False,
            "weight_promotion_permitted":False,
            "elapsed_s":max(time.perf_counter()-time_start,1e-9),
        }
    finally:
        with torch.no_grad():
            for p,orig in zip(params,originals,strict=True):
                p.copy_(orig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-unqualified-variance-counterfactual",action="store_true",required=True)
    p.add_argument("--seed",type=int,choices=NEW_SEEDS,required=True)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=run(seed=a.seed,repo_root=a.root,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps({"seed":a.seed,"status":report["status"],
                      "loss_forwards":report["total_model_loss_forwards"],
                      "results":{name:round(v["diagnostic_mean_loss_delta"],7)
                                 for name,v in report["results"].items()},
                      "saved_model_weights":0,"output":str(a.output)}))


if __name__=="__main__":
    main()
