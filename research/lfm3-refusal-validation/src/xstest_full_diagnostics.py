"""Purely descriptive XSTest lexical-proxy diagnostics; no safety classifier.

All model outputs stay private. Run only on exact frozen completed journal.
"""
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

from src.xstest_progress_audit import audit, snapshot_complete_lines
from src.paired_reviewed_effects import paired_interval
from src.xstest_450_eval import read_verified_dataset, ARM_ORDER


def diagnostics(rows, benchmark, iterations=2000):
    by_key={}
    for r in rows:
        key=(r["id"],r["arm"])
        if key in by_key:
            raise ValueError("duplicate generation")
        if r["id"] not in benchmark or r["arm"] not in ARM_ORDER:
            raise ValueError("unknown generation")
        if not isinstance(r.get("refusal_proxy"),bool):
            raise ValueError("missing lexical proxy boolean")
        by_key[key]=r
    if len(rows)!=1350 or len(by_key)!=1350 or len(benchmark)!=450:
        raise ValueError("requires exact 1350 distinct generations")
    if any((id, arm) not in by_key for id in benchmark for arm in ARM_ORDER):
        raise ValueError("incomplete three-checkpoint prompt pairs")
    output={"evidence_class":"descriptive, weak lexical keyword proxy",
            "human_safety_validated":False,
            "numerator_definition":"response contains pre-specified refusal-pattern keyword",
            "n_cases":450,"n_generations":1350,"per_stratum":{}}
    for group in ("safe","unsafe"):
        ids=[key for key, r in benchmark.items() if r["class"]==group]
        aggregates={}
        for arm in ARM_ORDER:
            signals=[by_key[(case,arm)]["refusal_proxy"] for case in ids]
            aggregates[arm]={"n":len(ids),"lexical_refusal_count":sum(signals),
                             "lexical_refusal_rate":round(sum(signals)/len(ids),6)}
        pairs={}
        for left,right in ((ARM_ORDER[0],ARM_ORDER[1]),(ARM_ORDER[0],ARM_ORDER[2]),(ARM_ORDER[1],ARM_ORDER[2])):
            from_to=Counter((by_key[(case,left)]["refusal_proxy"],by_key[(case,right)]["refusal_proxy"]) for case in ids)
            diffs=[int(by_key[(case,right)]["refusal_proxy"])-int(by_key[(case,left)]["refusal_proxy"]) for case in ids]
            intervals=paired_interval(diffs,seed=17,resamples=iterations)
            pairs[f"{right}_minus_{left}"]={
                "previously_no_now_yes":from_to[(False,True)],
                "previously_yes_now_no":from_to[(True,False)],
                "no_change_yes":from_to[(True,True)],
                "no_change_no":from_to[(False,False)],
                "difference":intervals,
            }
        output["per_stratum"][group]={"number_of_prompts":len(ids),"arms":aggregates,"paired_diagnostic_only":pairs}
    output["limitations"]=[
        "Lexical keywords do not validate harmful actionability, safe redirection, or refusal intent",
        "Single-turn English-only benchmark and max 128 generated tokens",
        "Model B direction estimated from just two calibration pairs",
        "Pretrained C lacks instruction tuning; comparisons are descriptive",
        "Human review NOT STARTED; cannot infer legitimate refusal or harmful compliance",
        "Per-prompt paired bootstrap confidence intervals describe only proxy variability, not grader uncertainty",
    ]
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    progress=audit(args.results_dir)
    if progress["recorded_generations"]!=1350 or progress["evidence_status"]!="GENERATIONS COMPLETE; HUMAN REVIEW PENDING":
        raise ValueError("full, validated 3-arm journal required")
    if progress["ignored_unterminated_tail"]:
        raise ValueError("unterminated response record: cannot finalize")
    records,partial,sha=snapshot_complete_lines(args.results_dir/"private_generations.jsonl")
    if partial:
        raise ValueError("partial record")
    benchmark={r["id"]:r for r in read_verified_dataset()}
    summary=diagnostics(records,benchmark)
    summary["private_journal_sha256"]=sha
    summary["benchmark_sha256"]=progress["benchmark_sha256"]
    summary["checkpoint_sha256"]=progress["checkpoint_sha256"]
    if args.output.exists():
        raise FileExistsError("refusing to overwrite prior evidence")
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
