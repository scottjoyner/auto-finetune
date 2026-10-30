"""Private two-reviewer annotation progress audit for three checkpoint XSTest.

Never uses model output to automatically grade safety, and never prints
prompt text, completion text, reviewer identities, or per-case judgments.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.xstest_450_eval import read_verified_dataset, ARM_ORDER
from src.blinded_xstest_review import RESPONSE_CATEGORIES, ACTIONABILITY


def classify_row(row, is_safe):
    if not row.get("reviewer_id","").strip():
        return "ungraded"
    answer=row.get("response_category","").strip()
    danger=row.get("actionable_harmful_content","").strip()
    useful=row.get("benign_helpfulness_0_3","").strip()
    if answer not in RESPONSE_CATEGORIES or danger not in ACTIONABILITY:
        return "ungraded"
    if answer=="unclear" or danger=="uncertain":
        return "requires_adjudication"
    if is_safe and useful not in ("0","1","2","3"):
        return "ungraded"
    if not is_safe and useful not in ("",):
        return "requires_adjudication"
    return "graded"


def load_review(path):
    with Path(path).open(newline="",encoding="utf-8") as file:
        records=list(csv.DictReader(file))
    if len(records)!=1350:
        raise ValueError("reviewer sheet must contain exactly 1350 records")
    ids=[r["review_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate review ID in sheet")
    for row in records:
        if row.get("response_category") not in RESPONSE_CATEGORIES+("",):
            raise ValueError("invalid category annotation")
        if row.get("actionable_harmful_content") not in ACTIONABILITY+("",):
            raise ValueError("invalid actionability annotation")
    return {r["review_id"]:r for r in records}


def audit_packets(review_dir, source=None):
    review_dir=Path(review_dir)
    lookup={r["id"]:r for r in (source if source is not None else read_verified_dataset())}
    mapping=json.loads((review_dir/"private_model_key.json").read_text())
    if len(mapping)!=1350 or len(lookup)!=450:
        raise ValueError("full XSTest packet must have 450 prompts x 3 models")
    case_arms=Counter()
    for key,identity in mapping.items():
        if not key.startswith("XR") or identity.get("source_id") not in lookup or identity.get("arm") not in ARM_ORDER:
            raise ValueError("invalid private packet identity")
        case_arms[(identity["source_id"],identity["arm"])] += 1
    if len(case_arms)!=1350 or any(v!=1 for v in case_arms.values()):
        raise ValueError("private mapping missing or duplicating a model/question")
    sheets=[load_review(review_dir/f"blinded_reviewer_{n}.csv") for n in (1,2)]
    if any(set(sheet)!=set(mapping) for sheet in sheets):
        raise ValueError("review sheet IDs do not match private mapping")
    states=[]
    names=[]
    for sheet in sheets:
        row_states=Counter()
        ids=set()
        for review_id,row in sheet.items():
            identity=mapping[review_id]
            state=classify_row(row,lookup[identity["source_id"]]["class"]=="safe")
            row_states[state]+=1
            if row.get("reviewer_id","").strip():
                ids.add(row["reviewer_id"].strip())
        if len(ids)>1:
            raise ValueError("one reviewer ID per sheet required")
        states.append(row_states)
        names.append(next(iter(ids)) if ids else None)
    if names[0] is not None and names[0]==names[1]:
        raise ValueError("two distinct reviewers required")
    disagreements=0
    if all(s["graded"]==1350 for s in states):
        for key in mapping:
            a,b=sheets[0][key],sheets[1][key]
            fields=("response_category","actionable_harmful_content")
            if lookup[mapping[key]["source_id"]]["class"]=="safe":
                fields+=("benign_helpfulness_0_3",)
            if any(a[f]!=b[f] for f in fields):
                disagreements+=1
    complete=(all(s["graded"]==1350 for s in states) and
              all(name for name in names) and names[0]!=names[1] and
              disagreements==0)
    return {
        "benchmark":"XSTest 450 prompts x 3 frozen checkpoints",
        "total_items_per_reviewer":1350,
        "reviewer_1":{"graded":states[0]["graded"],"ungraded":states[0]["ungraded"],
                      "needs_adjudication":states[0]["requires_adjudication"]},
        "reviewer_2":{"graded":states[1]["graded"],"ungraded":states[1]["ungraded"],
                      "needs_adjudication":states[1]["requires_adjudication"]},
        "disagreement_count_if_both_complete":disagreements if all(s["graded"]==1350 for s in states) else None,
        "full_independent_consensus_complete":bool(complete),
        "no_automatic_model_grading":True,
        "results_publication_approved":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--review-dir",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit_packets(a.review_dir),indent=2))


if __name__=="__main__":
    main()
