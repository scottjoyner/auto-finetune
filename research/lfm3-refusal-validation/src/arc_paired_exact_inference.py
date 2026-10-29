"""Exploratory exact matched-comparison sensitivity for the frozen ARC A/B/C study.

Reads only a published aggregate report. Never consumes or exports prompts,
answer keys, individual case scores, private human reviews, or model weights.
Scientific caveat: custom zero-shot likelihood scores, not official ARC scores.
"""
import argparse
import json
from math import comb
from pathlib import Path

from src.arc_full_likelihood_eval import ORDER

METRICS=("length_normalized_correct","raw_correct")
COMPARISONS=(
    (ORDER[0],ORDER[1]),
    (ORDER[0],ORDER[2]),
    (ORDER[1],ORDER[2]),
)


def exact_two_sided_discordant_p(gain, loss):
    """Exact conditional McNemar two-sided binomial p-value, no asymptotic shortcut."""
    if type(gain) is not int or type(loss) is not int or gain < 0 or loss < 0:
        raise ValueError("paired discordance counts must be nonnegative integers")
    n=gain+loss
    if n==0:
        return 1.0
    small=min(gain,loss)
    # Arbitrary-precision integers prevent underflow for 1,172 paired cases.
    numerator=2*sum(comb(n,i) for i in range(small+1))
    return min(1.0, numerator/(1<<n))


def holm_bonferroni(pvalues):
    """Monotone Holm familywise-adjusted p-values, preserving original order."""
    if not pvalues:
        raise ValueError("at least one p-value required")
    if any(not isinstance(p,(float,int)) or isinstance(p,bool) or not 0<=p<=1 for p in pvalues):
        raise ValueError("invalid p-value")
    m=len(pvalues)
    indices=sorted(range(m),key=lambda i:pvalues[i])
    adj=[None]*m
    running=0.0
    for i,index in enumerate(indices):
        running=max(running,(m-i)*pvalues[index])
        adj[index]=min(1.0,running)
    return adj


def evaluate(aggregate):
    if (aggregate.get("cases_per_model")!=1172
            or aggregate.get("total_scored")!=3516
            or aggregate.get("safety_evidence") is not False
            or aggregate.get("evidence_class")!="descriptive capability proxy, not official ARC score"):
        raise ValueError("requires exact frozen 1,172 x 3 custom ARC aggregate")
    if set(aggregate.get("checkpoint_sha256",{})) != set(ORDER) or \
            len(set(aggregate["checkpoint_sha256"].values()))!=3:
        raise ValueError("missing frozen three-arm checkpoint identities")
    if set(aggregate.get("metrics",{}))!=set(METRICS):
        raise ValueError("exactly two predetermined scoring protocols required")
    comparisons=[]
    for metric in METRICS:
        evidence=aggregate["metrics"][metric]
        by_model=evidence["by_checkpoint"]
        if set(by_model)!=set(ORDER):
            raise ValueError("model arm mismatch")
        for arm in ORDER:
            data=by_model[arm]
            if type(data.get("correct")) is not int or type(data.get("total")) is not int \
                    or data["total"]!=1172 or not 0<=data["correct"]<=1172:
                raise ValueError("invalid per-model accuracy count")
        paired=evidence["paired_comparisons"]
        expected={f"{right}_minus_{left}" for left,right in COMPARISONS}
        if set(paired)!=expected:
            raise ValueError("missing or additional comparison")
        for left,right in COMPARISONS:
            key=f"{right}_minus_{left}"
            record=paired[key]
            gained=record.get("became_correct")
            lost=record.get("became_incorrect")
            if any(type(v) is not int or v<0 for v in (gained,lost)):
                raise ValueError("invalid discordance")
            if gained+lost>1172 or gained-lost != by_model[right]["correct"]-by_model[left]["correct"]:
                raise ValueError("paired discordance does not agree with total accuracy")
            comparisons.append({
                "scoring":metric,
                "contrast":key,
                "n_questions":1172,
                "gained_correct":gained,
                "lost_correct":lost,
                "discordant_count":gained+lost,
                "percentage_point_difference":round(100*(gained-lost)/1172,4),
                "exact_two_sided_conditional_mcnemar_p":exact_two_sided_discordant_p(gained,lost),
            })
    adjusted=holm_bonferroni([row["exact_two_sided_conditional_mcnemar_p"] for row in comparisons])
    for row,adjusted_p in zip(comparisons,adjusted):
        row["holm_adjusted_p_across_six_exploratory_comparisons"]=adjusted_p
    return {
        "evidence_class":"exploratory paired exact conditional tests on custom capability scoring",
        "benchmark_source_sha256":aggregate.get("benchmark_sha256"),
        "checkpoint_sha256":aggregate["checkpoint_sha256"],
        "full_questions_per_checkpoint":1172,
        "number_of_checkpoints":3,
        "scoring_rules_evaluated":list(METRICS),
        "familywise_correction":"Holm-Bonferroni across 3 contrasts x 2 scoring conventions",
        "comparisons":comparisons,
        "safety_validated":False,
        "official_arc_benchmark":False,
        "limitations":[
            "Exploratory exact McNemar p-values assume matched questions and depend on evaluation design",
            "The two scoring conventions are correlated and strongly format sensitive",
            "Zero-shot custom answer-text likelihood is not an official ARC leaderboard score",
            "No equivalence or noninferiority margin was predeclared: failure to reject does not prove preservation",
            "Human-graded refusal, helpfulness and harmful actionability remain independent and incomplete",
            "The single edited checkpoint used only two calibration pairs",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    record=evaluate(json.loads(a.aggregate.read_text()))
    if a.output.exists():
        raise FileExistsError("cannot replace previous hypothesis-test evidence")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
