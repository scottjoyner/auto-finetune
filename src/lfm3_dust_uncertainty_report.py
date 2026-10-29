"""Compact, reproducible independent-summary view of new-seed Dust abstention.

Reruns the exact causal, before-measurement shrinkage decisions from raw
LFM3 real-model signed-direction receipts. An existing summary file is never
trusted without recomputing all source ledger checks, metrics and decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lfm3_dust_uncertainty_replay import SEEDS, replay, denied
from src.lfm3_lora_text_task_probe import dataset

SCHEMA="sensorgraft.lfm3.dust-uncertainty-seed456-summary.v1"


def build(reports):
    result=replay(reports)
    if result.get("status")!="exploratory_new_seed_prequential_replay":
        return {"schema":SCHEMA,"status":"blocked","reason":result.get("reason","invalid_sources"),
                "optimizer_switch_permitted":False,"weight_promotion_permitted":False}
    # Distinct operand/family identities across all prospective new-seed
    # training prompts; repeatedly probed +/- directions do not count as new
    # independent tasks.
    examples=[item["input_key"] for seed in SEEDS for item in dataset(seed)["train"][:12]]
    unique=len(set(examples))
    if unique<32:
        return {"schema":SCHEMA,"status":"blocked","reason":"insufficient_distinct_training_examples",
                "optimizer_switch_permitted":False,"weight_promotion_permitted":False}
    per_seed={}
    for seed in SEEDS:
        details=result["results"][str(seed)]
        per_seed[str(seed)]={
            "source_run_id":details["run_id"],
            "policies":{
                policy:{
                    "prequential_mse":v["prequential_mse"],
                    "zero_mse":v["zero_mse"],
                    "ratio_vs_zero":v["ratio_vs_zero"],
                    "probe_recommendations":v["probe_recommendations"],
                    "abstentions":v["abstentions"],
                    "final_shrinkage_coefficient":v["final_shrinkage_coefficient"],
                }
                for policy,v in details["policies"].items()
            },
        }
    aggregate=result["aggregate"]
    admitted=any(v["all_seed_mse_beats_zero_5pct"] and
                 v["total_recommended_probes"]>=1
                 for v in aggregate.values())
    # Even this hypothetical outcome merely suggests a new experimental
    # confirmation study and NEVER authorizes model/optimizer promotion.
    return {
        "schema":SCHEMA,
        "status":"hold_not_qualified" if not admitted else "separate_confirmation_study_only",
        "seeds":list(SEEDS),"source_train_episodes":36,
        "unique_training_example_keys":unique,
        "scored_future_episodes_per_policy":33,
        "scored_signed_directions_per_policy":264,
        "measured_perturbed_forwards":288,
        "measured_prompt_context_forwards":36,
        "model_backward_calls":0,"adapter_updates":0,
        "per_seed":per_seed,"aggregate":aggregate,
        "frozen_algorithm_preceded_new_seed_runs":True,
        "source_reports_independently_attested":False,
        "new_task_family_generalization_established":False,
        "gain_prediction_superiority_established":admitted,
        "model_quality_improvement_proven":False,
        "optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }


def verify(summary,reports):
    expected=build(reports)
    if expected["status"]=="blocked":
        return False
    return summary==expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only",action="store_true",required=True)
    p.add_argument("--evidence",nargs=3,type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if any(x.is_symlink() or not x.is_file() or x.stat().st_size>175000
           for x in args.evidence):
        p.error("untrusted_evidence_paths")
    source=[json.loads(x.read_text(encoding="utf-8")) for x in args.evidence]
    result=build(source)
    if result["status"]=="blocked":
        p.error(result["reason"])
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({"status":result["status"],"aggregate":result["aggregate"],
                      "output":str(args.output),"weight_promotion_permitted":False}))
if __name__=="__main__":
    main()
