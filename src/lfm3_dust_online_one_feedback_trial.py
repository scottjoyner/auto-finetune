"""Real LFM3 four-episode online one-feedback head experiment, no backprop.

Three PRE-REGISTERED seeds 28/29/30, four fresh public-text episodes each,
one measured signed model-probe per candidate-choice policy per episode.
Compare fixed random, frozen historical LFM3 neural Q/K attention, and the
same 195-weight Q/K head adapting by two head-only derivative-free proposals
to the PREVIOUS neural-selected train gain. No current gain can alter the
current forecast; independent six-prompt diagnostics are never fed to head.
Identical 2,048 LoRA-B adapter starting hash and temporary perturbation
scope for every episode/arm; all model mutations reverted in finally blocks.
All research source receipts unsigned; no model weights exported.
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
from src.lfm3_dust_equal_probe_benchmark import _diag,digest
from src.lfm3_dust_text_family_curriculum import WORDS,_one,curriculum
from src.lfm3_dust_text_family_transfer_review import review as source_review
from src.lfm3_dust_task_shift_probe import training_source_path,rebuild_training_head
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_matched_gradient_probe import select_b_only
from src.lfm3_dust_online_single_feedback import OnlineSingleFeedbackHead
from src.lfm3_real_lora_probe import BASE_SHA256,MODEL_PATH,_load,_preflight,_sha

SCHEMA="sensorgraft.lfm3.dust-online-single-feedback-trial.v1"
SEEDS=(28,29,30)
HEAD_SEEDS=(10,11,12)
EPISODES=(12,13,14)
POLICIES=("random_one_probe","frozen_head_one_probe","adaptive_head_one_probe")
SIGMA=.05


def tasks(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered_online_lfm3_seed")
    old=curriculum(HEAD_SEEDS[SEEDS.index(seed)])
    used={x["input_key"] for rows in old.values() for x in rows}
    rng=random.Random(2026103100+seed)
    def make(family):
        for _ in range(2000):
            a,b=rng.sample(WORDS,2)
            item=_one(family,a,b)
            if item["input_key"] not in used:
                used.add(item["input_key"])
                return item
        raise RuntimeError("duplicate_unseen_diagnostic_prompts")
    episodes=[]
    for ep in EPISODES:
        episodes.append({
            "episode":ep,
            "training":[make("first_word" if ep%2==0 else "last_word")],
            "diagnostic_same_family":[make("first_word"),make("last_word"),
                                      make("first_word"),make("last_word")],
            "diagnostic_transfer":[make("uppercase"),make("uppercase")],
        })
    assert len(used)>=21
    return episodes


def candidates(seed,episode):
    if (type(seed) is not int or seed not in SEEDS
            or type(episode) is not int or episode not in EPISODES):
        raise ValueError("unregistered_direction_episode")
    rng=random.Random(6631300+seed*30+episode)
    directions=[]
    for i in range(4):
        v=list(_unit([rng.gauss(0,1) for _ in range(8)]))
        directions.extend([
            {"pair":i,"sign":1,"direction":v},
            {"pair":i,"sign":-1,"direction":[-x for x in v]},
        ])
    return directions


def policy_choice(seed,episode,frozen,adaptive):
    if (type(seed) is not int or seed not in SEEDS
            or type(episode) is not int or episode not in EPISODES
            or len(frozen)!=8 or len(adaptive)!=8
            or any(type(x) not in (float,int) or not math.isfinite(x)
                   for x in list(frozen)+list(adaptive))):
        raise ValueError("wrong_preprobe_rank")
    return {
        "random_one_probe":random.Random(64001+seed*100+episode).randrange(8),
        "frozen_head_one_probe":max(range(8),key=lambda i:(frozen[i],-i)),
        "adaptive_head_one_probe":max(range(8),key=lambda i:(adaptive[i],-i)),
    }


def run(*,seed,repo_root,model_path=MODEL_PATH):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered_model_trial")
    old=HEAD_SEEDS[SEEDS.index(seed)]
    path=training_source_path(Path(repo_root),old)
    if (path.is_symlink() or not path.is_file() or
            not 100<path.stat().st_size<=175000):
        raise RuntimeError("missing_authentic_historical_head")
    original_bytes=path.read_bytes()
    source=json.loads(original_bytes)
    if source_review(source).get("status")!="manual_transfer_review_only":
        raise RuntimeError("original_head_training_not_reproducible")
    frozen=rebuild_training_head(source)
    online=OnlineSingleFeedbackHead(rebuild_training_head(source),seed=seed)
    if (frozen.parameter_sha256!=online.head.parameter_sha256
            or FrozenTaskShiftGate(source=source).verified_skill):
        raise RuntimeError("incorrect_old_head_or_unqualified_training_scope")
    frozen_sha=frozen.parameter_sha256
    initial_online_sha=online.head.parameter_sha256
    episodes=tasks(seed)
    task_digests=[{"episode":row["episode"],
        "training":digest(row["training"]),
        "diagnostic_same_family":digest(row["diagnostic_same_family"]),
        "diagnostic_transfer":digest(row["diagnostic_transfer"])}
        for row in episodes]
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
    adapter_initial=_sha(*params)
    basis=_basis(torch,params,seed=old)
    if sum(p.numel() for p in params)!=2048:
        raise RuntimeError("wrong_adapter_scope")

    def temporary(d,call):
        blocks=_direction(torch,basis,d)
        norm=math.sqrt(sum(float((SIGMA*x).square().sum()) for x in blocks))
        if not 0<norm<=.15:
            raise RuntimeError("unsafe_lora_perturbation")
        with torch.no_grad():
            for p,base,offset in zip(params,originals,blocks,strict=True):
                p.copy_(base+SIGMA*offset)
        try:
            return call()
        finally:
            with torch.no_grad():
                for p,base in zip(params,originals,strict=True):
                    p.copy_(base)
            if _sha(*params)!=adapter_initial:
                raise RuntimeError("adapter_exact_restoration_failed")
    report=[]
    start=time.perf_counter()
    try:
        for row in episodes:
            episode=row["episode"]
            step_start=time.perf_counter()
            train=row["training"][0]
            head_start=time.perf_counter()
            tune=online.prepare(episode=episode,adaptive=True)
            head_tuning_seconds=time.perf_counter()-head_start
            context_start=time.perf_counter()
            context=prompt_only_activation_context(torch,model,tokenizer,
                                                   train["prompt"])
            context_seconds=time.perf_counter()-context_start
            proposal=candidates(seed,episode)
            pred_start=time.perf_counter()
            fixed_preds=[frozen.predict(x["direction"],context,episode=episode)[
                "predicted_gain"] for x in proposal]
            dynamic_preds=online.forecast(
                [x["direction"] for x in proposal],context,episode=episode)
            forecast_seconds=time.perf_counter()-pred_start
            chosen=policy_choice(seed,episode,fixed_preds,dynamic_preds)
            clean=_loss(torch,model,tokenizer,train)
            base=_diag(torch,model,tokenizer,row)
            arms={}
            # All three choices and both head prediction arrays were
            # frozen BEFORE measuring any gain from the current episode.
            for policy in POLICIES:
                index=chosen[policy]
                candidate=proposal[index]
                t0=time.perf_counter()
                perturbed=temporary(candidate["direction"],
                    lambda:_loss(torch,model,tokenizer,train))
                gain=clean-perturbed
                accepted=gain>1e-8
                if policy=="adaptive_head_one_probe":
                    # Exactly one TRAIN-only label advances Q/K head memory;
                    # it cannot alter forecasts for this already-chosen step.
                    observation=online.observe(candidate["direction"],context,
                                               gain,episode=episode)
                else:
                    observation=None
                after=(temporary(candidate["direction"],
                       lambda:_diag(torch,model,tokenizer,row))
                       if accepted else _diag(torch,model,tokenizer,row))
                arms[policy]={
                    "selected_index":index,
                    "train_gain":gain,"train_clean_loss":clean,
                    "train_perturbed_loss":perturbed,
                    "training_gain_positive":accepted,
                    "diagnostic_after":after,
                    "mean_diagnostic_loss_delta":(
                        after["mean_completion_loss"]-base["mean_completion_loss"]),
                    "same_family_loss_delta":(
                        after["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                        base["by_family"]["diagnostic_same_family"]["mean_completion_loss"]),
                    "transfer_loss_delta":(
                        after["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                        base["by_family"]["diagnostic_transfer"]["mean_completion_loss"]),
                    "model_probe_forwards":1,"model_diagnostic_forwards":6,
                    "model_backwards":0,
                    "one_train_only_observation":observation,
                    "policy_elapsed_seconds":time.perf_counter()-t0,
                    "adapter_sha_after_restoring":_sha(*params),
                    "checkpoint_saved":False,
                    "production_execution_permitted":False,
                }
            if (frozen.parameter_sha256!=frozen_sha
                    or _sha(*params)!=adapter_initial or
                    any(p.grad is not None or p.requires_grad
                        for p in model.parameters())):
                raise RuntimeError("model_backprop_or_frozen_head_mutated")
            report.append({
                "episode":episode,"task_manifests":task_digests[episode-12],
                "preprobe_context":list(context),
                "signed_directions":proposal,
                "frozen_forecasts_before_any_probe":fixed_preds,
                "adaptive_forecasts_before_any_probe":dynamic_preds,
                "candidate_policy_choice":chosen,
                "training_only_head_update_before_current_forecasts":tune,
                "frozen_head_sha256":frozen.parameter_sha256,
                "adaptive_head_sha_after_training_only":online.head.parameter_sha256,
                "baseline_diagnostic":base,
                "clean_training_loss":clean,"policy_results":arms,
                "head_training_seconds":head_tuning_seconds,
                "context_model_forward_seconds":context_seconds,
                "both_heads_forecast_seconds":forecast_seconds,
                "episode_inference_elapsed_s":time.perf_counter()-step_start,
                "model_loss_forwards":28,
                "prompt_context_model_forwards":1,
                "model_backward_count":0,
                "persistent_model_updates":0,
            })
            print(json.dumps({
                "seed":seed,"episode":episode,
                "qk_training_updates":tune["head_only_accepted_updates"],
                "qk_head_only_objectives":tune["head_only_objective_forwards"],
                "random_delta":arms["random_one_probe"]["mean_diagnostic_loss_delta"],
                "frozen_delta":arms["frozen_head_one_probe"]["mean_diagnostic_loss_delta"],
                "adaptive_delta":arms["adaptive_head_one_probe"]["mean_diagnostic_loss_delta"],
            },allow_nan=False),flush=True)
        if (online.total_head_only_objective_forwards>10
                or online.total_head_accepted_updates>4
                or frozen.parameter_sha256!=frozen_sha
                or _sha(*params)!=adapter_initial):
            raise RuntimeError("unexpected_head_training_budget_or_persistent_adapter")
        return {
            "schema":SCHEMA,
            "status":"offline_online_one_label_head_research_only",
            "seed":seed,"source_head_seed":old,"run_id":uuid.uuid4().hex,
            "source_head_receipt_sha256":hashlib.sha256(original_bytes).hexdigest(),
            "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE_SHA256,
            "frozen_start_head_sha256":frozen_sha,
            "frozen_end_head_sha256":frozen.parameter_sha256,
            "online_start_head_sha256":initial_online_sha,
            "online_end_head_sha256":online.head.parameter_sha256,
            "online_head_parameters":195,
            "online_head_l2_drift":online.head.update_norm,
            "online_training_gain_observations":3,
            "online_head_only_objective_forwards":online.total_head_only_objective_forwards,
            "online_head_only_accepted_updates":online.total_head_accepted_updates,
            "trained_head_uses_diagnostic_feedback":False,
            "model_parameters_eligible_for_temporary_step":2048,
            "model_adapter_initial_sha256":adapter_initial,
            "model_adapter_final_sha256":_sha(*params),
            "sample_manifests":task_digests,
            "total_episodes":3,"policy_names":list(POLICIES),
            "candidate_model_probes_per_policy_per_episode":1,
            "diagnostic_model_forwards_per_policy_per_episode":6,
            "total_model_completion_loss_forwards":84,
            "total_model_prompt_context_forwards":3,
            "training_model_backward_calls":0,
            "persistent_model_adapter_updates":0,
            "model_checkpoints_saved":0,
            "selection_test_partition_opened":False,
            "head_proposals_are_backprop_free":True,
            "model_source_custody_independently_attested":False,
            "model_inference_total_elapsed_s":time.perf_counter()-start,
            "compute_budget_equal_in_training_probes_but_not_total":True,
            "model_update_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
            "episodes":report,
        }
    finally:
        with torch.no_grad():
            for p,base in zip(params,originals,strict=True):
                p.copy_(base)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--allow-one-label-head-learning-counterfactual",
                   action="store_true",required=True)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path",type=Path,default=Path(MODEL_PATH))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error("evidence output exists; refusing to load model/overwrite a run")
    result=run(seed=a.seed,repo_root=a.root,model_path=a.model_path)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps({"seed":a.seed,"status":result["status"],
                      "qk_updates":result["online_head_only_accepted_updates"],
                      "one_probe_policy_episodes":result["total_episodes"],
                      "adapter_saved":0,"output":str(a.output)},allow_nan=False))


if __name__=="__main__":
    main()
