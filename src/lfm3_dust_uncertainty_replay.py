"""Frozen-algorithm, new-seed prequential evaluation of Dust shrinkage veto.

The base LFM3 model was actually run in the strict all-candidate forward-only
audit. This deterministic replay recomputes online gate decisions from stored
BEFORE-probe forecasts and AFTER-probe scalar outcomes, one episode at a time.
It is not equivalent to an online deployed controller, does not perform model
updates, and must never be described as independent real-task generalization.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from src.lfm3_dust_prequential_review import review as review_source
from src.lfm3_dust_uncertainty_gate import CausalShrinkageGate

SCHEMA = "sensorgraft.lfm3.dust-shrinkage-prequential-newseed.v1"
POLICIES = ("direction", "activation", "calibrated")
SEEDS = (4,5,6)
EPISODES = 12


def denied(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "model_update_permitted": False, "optimizer_switch_permitted": False,
            "weight_promotion_permitted": False}


def replay(receipts):
    if not isinstance(receipts, list) or len(receipts)!=3:
        return denied("exact_three_new_seed_receipts_required")
    if sorted(x.get("seed",-1) for x in receipts if isinstance(x,dict)) != list(SEEDS):
        return denied("unseen_new_seed_cohort_required")
    if len({r.get("model_run_id") for r in receipts}) != 3:
        return denied("nonunique_real_model_receipt")
    outputs={}
    for report in sorted(receipts,key=lambda r:r["seed"]):
        check=review_source(report)
        if check.get("status")!="descriptive_review_required" or check["training_episodes"]!=EPISODES:
            return denied("source_model_receipt_failed")
        policies={}
        for policy in POLICIES:
            gate=CausalShrinkageGate()
            episodes=[]
            for episode,row in enumerate(report["episode_receipts"]):
                cases=row["candidates"]
                if len(cases)!=8:
                    return denied("candidate_coverage_shortfall")
                past=[gate.predict(c["predictions"][policy],episode=episode) for c in cases]
                # These are preprobe forecasts. Candidate measurements below
                # enter learning only AFTER all eight decisions are frozen.
                gains=[c["measured_gain"] for c in cases]
                eligible=[i for i,p in enumerate(past) if p["eligible_to_probe"]]
                selected=(max(eligible,key=lambda i:(past[i]["conservative_lower_bound"],-i))
                          if eligible else None)
                if selected is not None and (past[selected]["conservative_lower_bound"]<=0
                                             or past[selected]["prior_training_episodes"]<4):
                    return denied("unproven_admission_decision")
                episodes.append({
                    "episode":episode, "head":policy,
                    "prior_episode_count":gate.episodes,
                    "before_coefficient":gate.coefficient,
                    "candidate_forecasts":[p["predicted_gain"] for p in past],
                    "candidate_raw_forecasts":[p["raw_prediction"] for p in past],
                    "candidate_measured_gains":gains,
                    "eligible_candidate_indices":eligible,
                    "selected_candidate":selected,
                    "measured_gain_if_selected":gains[selected] if selected is not None else None,
                    "zero_baseline_gain_if_selected":gains[0],
                    "zero_baseline_mse":statistics.mean(g*g for g in gains),
                    "calibrated_mse":statistics.mean(
                        (p["predicted_gain"]-g)**2
                        for p,g in zip(past,gains,strict=True)),
                    "raw_mse":statistics.mean(
                        (p["raw_prediction"]-g)**2
                        for p,g in zip(past,gains,strict=True)),
                    "number_of_new_model_forwards_due_to_replay":0,
                })
                gate.observe_episode(
                    [(cases[i]["predictions"][policy],gains[i]) for i in range(8)],
                    episode=episode,split="train",measured=True)
            evaluated=episodes[1:]
            if len(evaluated)!=11:
                return denied("wrong_evaluation_horizon")
            mse=statistics.mean(x["calibrated_mse"] for x in evaluated)
            zero=statistics.mean(x["zero_baseline_mse"] for x in evaluated)
            n=sum(e["selected_candidate"] is not None for e in evaluated)
            policies[policy]={
                "scored_episodes":len(evaluated),"scored_direction_predictions":88,
                "prequential_mse":mse, "zero_mse":zero,
                "ratio_vs_zero":mse/zero if zero>0 else math.inf,
                "mean_raw_mse":statistics.mean(x["raw_mse"] for x in evaluated),
                "abstentions":len(evaluated)-n,"probe_recommendations":n,
                "mean_gain_when_recommended":(
                    statistics.mean(x["measured_gain_if_selected"] for x in evaluated
                                    if x["selected_candidate"] is not None)
                    if n else None),
                "final_shrinkage_coefficient":gate.coefficient,
                "episode_decisions":episodes,
            }
        outputs[str(report["seed"])]={"run_id":report["model_run_id"],"policies":policies}
    summary={}
    for policy in POLICIES:
        entries=[outputs[str(s)]["policies"][policy] for s in SEEDS]
        summary[policy]={
            "mean_seedwise_ratio_vs_zero":statistics.mean(v["ratio_vs_zero"] for v in entries),
            "all_seed_mse_beats_zero_5pct":all(v["ratio_vs_zero"]<=.95 for v in entries),
            "total_recommended_probes":sum(v["probe_recommendations"] for v in entries),
            "total_abstentions":sum(v["abstentions"] for v in entries),
            "total_scored_predictions":sum(v["scored_direction_predictions"] for v in entries),
        }
    return {
        "schema":SCHEMA,"status":"exploratory_new_seed_prequential_replay",
        "model":"LiquidAI/LFM3-1.7B","seeds":list(SEEDS),
        "policy_algorithm_frozen_before_new_seed_model_runs":True,
        "new_seed_real_model_episodes":3*EPISODES,
        "original_source_model_adapter_updates":0,
        "source_model_final_test_evaluated":False,
        "new_model_forwards_in_replay":0,
        "results":outputs,"aggregate":summary,
        "independent_real_task_generalization_proven":False,
        "independent_execution_attestation":False,
        "optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--new-seed-evidence",nargs=3,required=True,type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        files=args.new_seed_evidence
        if any(x.is_symlink() or not x.is_file() or x.stat().st_size>175000 for x in files):
            raise ValueError("unsafe_source")
        result=replay([json.loads(p.read_text(encoding="utf-8")) for p in files])
    except (ValueError,OSError,UnicodeError,OverflowError):
        result=denied("unreadable_evidence")
    if result["status"]=="blocked":
        raise SystemExit(result["reason"])
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({"status":result["status"],"aggregate":result["aggregate"],
                      "output":str(args.output),"weight_promotion_permitted":False}))
if __name__=="__main__":
    main()
