"""Inter-rater reliability for completed blinded three-arm XSTest reviews.

Never performs automatic safety grading. Produces ONLY aggregate metrics,
without reviewer names, prompt text, model response text or model mappings.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.reviewed_xstest_metrics import read_review
from src.reviewer_progress_audit import classify_row
from src.xstest_450_eval import read_verified_dataset, ARM_ORDER


def cohen_kappa(pairs, categories, quadratic=False):
    """Kappa, returning None when all possible variation vanishes."""
    if not pairs:
        raise ValueError("nonempty paired ratings required")
    if any(a not in categories or b not in categories for a,b in pairs):
        raise ValueError("out-of-vocabulary rating")
    n=len(pairs)
    ca=Counter(a for a,b in pairs)
    cb=Counter(b for a,b in pairs)
    scale=max(1,len(categories)-1)
    idx={c:i for i,c in enumerate(categories)}
    def weight(a,b):
        d=idx[a]-idx[b]
        return (d/scale)**2 if quadratic else int(d!=0)
    observed=sum(weight(a,b) for a,b in pairs)/n
    expected=sum(weight(a,b)*ca[a]*cb[b] for a in categories for b in categories)/(n*n)
    if expected <= 1e-12:
        return None
    return round(1-observed/expected,6)


def measure(first,second,mapping,benchmark):
    if len(mapping)!=1350 or set(first)!=set(second) or set(first)!=set(mapping):
        raise ValueError("requires exact same 1350 blinded case identities")
    if len(benchmark)!=450:
        raise ValueError("requires complete 450-item XSTest labels")
    validated=set()
    graders=[set(),set()]
    category_pairs=[]
    actionability_pairs=[]
    helpful_pairs=[]
    for rid in sorted(mapping):
        m=mapping[rid]
        qid=m.get("source_id")
        arm=m.get("arm")
        if qid not in benchmark or arm not in ARM_ORDER or (qid,arm) in validated:
            raise ValueError("invalid or duplicated prompt/checkpoint mapping")
        validated.add((qid,arm))
        is_safe=benchmark[qid]["class"]=="safe"
        a,b=first[rid],second[rid]
        for idx,r in enumerate((a,b)):
            if classify_row(r,is_safe)!="graded":
                raise ValueError("incomplete or uncertain human review")
            grader=r["reviewer_id"].strip()
            if not grader:
                raise ValueError("reviewer identifier missing")
            graders[idx].add(grader)
        category_pairs.append((a["response_category"],b["response_category"]))
        actionability_pairs.append((a["actionable_harmful_content"],b["actionable_harmful_content"]))
        if is_safe:
            helpful_pairs.append((a["benign_helpfulness_0_3"],b["benign_helpfulness_0_3"]))
    if len(validated)!=1350 or any(len(g)!=1 for g in graders) or graders[0]==graders[1]:
        raise ValueError("distinct, consistent reviewers and complete triplets required")

    def reliability(pairs,labels,quadratic=False):
        exact=sum(a==b for a,b in pairs)
        return {
            "paired_items":len(pairs),
            "exact_agreement_count":exact,
            "disagreement_count":len(pairs)-exact,
            "exact_agreement_rate":round(exact/len(pairs),6),
            "cohens_kappa":cohen_kappa(pairs,labels,quadratic),
            "kappa_undefined_if_no_variation":True,
        }
    category=reliability(category_pairs,("full_compliance","partial_refusal","full_refusal"))
    action=reliability(actionability_pairs,("yes","no"))
    helpful=reliability(helpful_pairs,("0","1","2","3"),quadratic=True)
    # Count item-level disagreement across all applicable fields, including
    # safe helpfulness, independently of how the private mapping is ordered.
    disagreement_ids = set()
    for rid in mapping:
        qid=mapping[rid]["source_id"]
        a,b=first[rid],second[rid]
        if (a["response_category"] != b["response_category"] or
            a["actionable_harmful_content"] != b["actionable_harmful_content"] or
            (benchmark[qid]["class"]=="safe" and
             a["benign_helpfulness_0_3"] != b["benign_helpfulness_0_3"])):
            disagreement_ids.add(rid)
    return {
        "study":"XSTest exactly three frozen checkpoints, two blind human reviewers",
        "fully_reviewed_response_pairs":len(mapping),
        "human_review_only":True,
        "reviewer_names_in_report":False,
        "checkpoint_level_results_revealed":False,
        "response_category":category,
        "actionable_harmful_content":action,
        "safe_helpfulness_quadratic_weighted":helpful,
        "cases_requiring_adjudication":len(disagreement_ids),
        "independent_consensus_complete":len(disagreement_ids)==0,
        "safety_release_approved":False,
        "notes":[
            "Agreement does not establish validity against ground truth",
            "Zero variation makes kappa undefined even when exact agreement is 100 percent",
            "Kappa estimates depend on response prevalence and reviewer rubric",
            "Keep original review sheets immutable and retain private adjudication audit trail"
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--review-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    root=args.review_dir
    first=read_review(root/"blinded_reviewer_1.csv")
    second=read_review(root/"blinded_reviewer_2.csv")
    mapping=json.loads((root/"private_model_key.json").read_text())
    benchmark={r["id"]:r for r in read_verified_dataset()}
    report=measure(first,second,mapping,benchmark)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite previous reliability report")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    args.output.chmod(0o600)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
