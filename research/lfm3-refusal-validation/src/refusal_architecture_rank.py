"""Report refusal-keyword signals by frozen LFM3 architecture and checkpoint.

These are NOT actual refusal judgments and do not justify modifying safety.
The three arms share the same architecture, so differences cannot be
attributed to architectural layers.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from src.official_arc_readiness import ARMS_EXPECTED
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_progress_audit import audit

EXPECTED_COUNTS={"safe":250,"unsafe":200}
EXPECTED_LAYERS={"conv":13,"sliding_attention":5,"full_attention":6}


def wilson(k,n,z=1.959963984540054):
    p=k/n
    d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [0.0 if k==0 else max(0.,mid-half),1.0 if k==n else min(1.,mid+half)]


def summarize(proxy,progress,configs,hashes):
    if (proxy.get("evidence_class")!="descriptive, weak lexical keyword proxy"
        or proxy.get("human_safety_validated") is not False
        or proxy.get("n_cases")!=450 or proxy.get("n_generations")!=1350
        or progress.get("recorded_generations")!=1350
        or progress.get("journal_integrity")!="passed"
        or progress.get("manual_review")!="NOT STARTED"
        or progress.get("ignored_unterminated_tail")
        or proxy.get("private_journal_sha256")!=progress.get("journal_complete_lines_sha256")
        or proxy.get("benchmark_sha256")!=progress.get("benchmark_sha256")
        or proxy.get("checkpoint_sha256")!=progress.get("checkpoint_sha256")
        or hashes!=proxy.get("checkpoint_sha256")):
        raise ValueError("not a complete, integrity-pinned ungraded XSTest proxy")
    if tuple(hashes)!=ARMS_EXPECTED or len(set(hashes.values()))!=3:
        raise ValueError("must use the exact three frozen checkpoint identities")
    if tuple(configs)!=ARMS_EXPECTED:
        raise ValueError("not the original three model configurations")
    reference=configs[ARMS_EXPECTED[0]]
    layers=reference.get("layer_types")
    if (not isinstance(layers,list) or len(layers)!=24 or
        dict(Counter(layers))!=EXPECTED_LAYERS or
        reference.get("hidden_size")!=2048 or
        reference.get("num_attention_heads")!=16 or
        any(configs[arm].get("layer_types")!=layers or
            configs[arm].get("hidden_size")!=2048 or
            configs[arm].get("num_attention_heads")!=16
            for arm in ARMS_EXPECTED)):
        raise ValueError("different or unrecognized hybrid architecture in study")
    if set(proxy.get("per_stratum",{}))!=set(EXPECTED_COUNTS):
        raise ValueError("missing source strata")
    strata={}
    for name,total in EXPECTED_COUNTS.items():
        data=proxy["per_stratum"][name]
        if data.get("number_of_prompts")!=total or tuple(data.get("arms",{}))!=ARMS_EXPECTED:
            raise ValueError("incorrect per-stratum model coverage")
        arms={}
        for arm in ARMS_EXPECTED:
            measure=data["arms"][arm]
            k=measure.get("lexical_refusal_count")
            fraction=measure.get("lexical_refusal_rate")
            if (measure.get("n")!=total or
                type(k) is not int or not 0<=k<=total or
                type(fraction) not in (float,int) or not math.isfinite(fraction) or
                abs(k/total-fraction)>1e-10):
                raise ValueError("invalid lexical refusal indicator count")
            arms[arm]={
                "keyword_positive":k,"n":total,
                "fraction":k/total,
                "wilson95_exploratory":[round(v,6) for v in wilson(k,total)],
            }
        max_count=max(v["keyword_positive"] for v in arms.values())
        strata[name]={
            "n":total,"arms":arms,
            "highest_keyword_positive_arm":[a for a,v in arms.items()
                                          if v["keyword_positive"]==max_count],
            "highest_keyword_positive_count":max_count,
        }
    return {
        "evidence_class":"lexical proxy ranking; not actual or human-verified refusal",
        "model_architecture":"LFM3-1.7B identical hybrid layers across frozen A/B/C",
        "architecture_summary":{
            "decoder_layers":len(layers),"layer_types":dict(Counter(layers)),
            "layer_type_sequence":layers,"hidden_size":2048,"attention_heads":16,
            "architecture_changed_across_arms":False,
        },
        "checkpoint_sha256":hashes,
        "private_journal_sha256":proxy["private_journal_sha256"],
        "strata":strata,
        "human_rated_refusal_rate_available":False,
        "human_review_complete":False,
        "claim_any_specific_layer_causes_refusal":False,
        "zero_refusal_established":False,
        "model_release_approved":False,
        "limitations":[
            "The highest lexical indicator is not necessarily the highest actual refusal rate",
            "High unsafe-prompt refusal can be appropriate; high benign-prompt refusal may indicate over-refusal",
            "All three checkpoints share exactly the same hybrid architecture; only training/weight state differs",
            "Observed differences do not prove that a particular convolution or attention layer controls refusal",
            "Human reviewers have not labeled the 1350 private generations",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--proxy",required=True,type=Path)
    p.add_argument("--results-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("read-only ranking may not replace earlier evidence")
    proxy=json.loads(args.proxy.read_text())
    configs={arm:json.loads((ARMS[arm]/"config.json").read_text())
             for arm in ARMS_EXPECTED}
    report=summarize(proxy,audit(args.results_dir),configs,checked_arm_paths(ARMS))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"highest_benign_keyword_arm":report["strata"]["safe"]["highest_keyword_positive_arm"],
        "highest_unsafe_keyword_arm":report["strata"]["unsafe"]["highest_keyword_positive_arm"],
        "verified_human_refusals":False,"output":str(args.out)},indent=2))

if __name__=="__main__":
    main()
