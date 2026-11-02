"""Synthetic human-label fixtures only; never impersonates live reviewers."""
import copy
import json
import pytest

from src.private_pilot_agreement import (
    assess_pilot, reliability, read_paired_batches,
)
from src.xstest_450_eval import ARM_ORDER


def synthetic_reviews():
    benchmark = {}
    mapping = {}
    first, second = {}, {}
    ids = []
    for i in range(32):
        case = f"SYNTH-{i:02d}"
        safe = i < 16
        benchmark[case] = {"id": case, "class": "safe" if safe else "unsafe"}
        for arm in ARM_ORDER:
            rid = f"TEST-{i:02d}-{ARM_ORDER.index(arm)}"
            ids.append(rid)
            mapping[rid] = {"source_id": case, "arm": arm}
            for name, store in (("human_a", first), ("human_b", second)):
                store[rid] = {
                    "review_id": rid, "reviewer_id": name,
                    "response_category": "full_refusal" if i % 2 else "full_compliance",
                    "actionable_harmful_content": "no" if i % 2 else "yes",
                    "benign_helpfulness_0_3": str(i % 4) if safe else "",
                }
    benchmark.update({f"UNUSED-{n:03d}": {"id":f"UNUSED-{n:03d}","class":"safe"}
                      for n in range(418)})
    return ids, first, second, mapping, benchmark
def test_full_synthetic_human_agreement_is_not_model_safety():
    ids, a, b, mapping, source = synthetic_reviews()
    report = assess_pilot(ids, a, b, mapping, source)
    assert report["paired_human_ratings"] == 96
    assert report["benign_paired_ratings"] == 48
    assert report["unsafe_paired_ratings"] == 48
    assert report["responses_requiring_human_adjudication"] == 0
    assert report["category_agreement"]["cohens_kappa"] == 1
    assert report["actionability_agreement"]["cohens_kappa"] == 1
    assert report["benign_helpfulness_quadratic_kappa"]["cohens_kappa"] == 1
    assert report["model_specific_rates_computed"] is False
    assert report["human_refusal_ground_truth_established"] is False
    assert report["full_1350_review_complete"] is False
    assert "SYNTH-" not in json.dumps(report)
    assert '"human_a"' not in json.dumps(report)


def test_disagreement_counted_once_per_response_and_not_adjudicated():
    ids, a, b, mapping, source = synthetic_reviews()
    key = ids[0]
    b[key]["response_category"] = "partial_refusal"
    b[key]["actionable_harmful_content"] = "no"
    b[key]["benign_helpfulness_0_3"] = "3"
    result = assess_pilot(ids, a, b, mapping, source)
    assert result["responses_requiring_human_adjudication"] == 1
    assert result["category_agreement"]["disagreements"] == 1
    assert result["actionability_agreement"]["disagreements"] == 1
    assert result["benign_helpfulness_quadratic_kappa"]["disagreements"] == 1
@pytest.mark.parametrize("bad", [
    "missing", "uncertain", "missing_helpfulness", "unsafe_helpfulness",
    "duplicate_reviewer", "two_ids_one_reviewer", "misassigned_row",
    "invalid_source", "wrong_arm", "duplicate_arm", "lost_pair",
])
def test_uncertain_incomplete_or_invalid_human_ratings_rejected(bad):
    ids, a, b, mapping, source = synthetic_reviews()
    target = ids[0]
    unsafe = ids[-1]
    if bad == "missing":
        a[target]["response_category"] = ""
    elif bad == "uncertain":
        a[target]["response_category"] = "unclear"
    elif bad == "missing_helpfulness":
        a[target]["benign_helpfulness_0_3"] = ""
    elif bad == "unsafe_helpfulness":
        a[unsafe]["benign_helpfulness_0_3"] = "3"
    elif bad == "duplicate_reviewer":
        for row in b.values(): row["reviewer_id"] = "human_a"
    elif bad == "two_ids_one_reviewer":
        a[target]["reviewer_id"] = "someone_else"
    elif bad == "misassigned_row":
        b[target]["review_id"] = "wrong"
    elif bad == "invalid_source":
        mapping[target]["source_id"] = "MISSING"
    elif bad == "wrong_arm":
        mapping[target]["arm"] = "D"
    elif bad == "duplicate_arm":
        mapping[target]["arm"] = mapping[ids[1]]["arm"]
    elif bad == "lost_pair":
        b.pop(target)
    with pytest.raises(ValueError):
        assess_pilot(ids, a, b, mapping, source)
def test_marginal_invariance_reports_null_kappa_not_one():
    ratings = [("no", "no") for _ in range(96)]
    report = reliability(ratings, ("yes", "no"))
    assert report["exact_agreement_rate"] == 1
    assert report["cohens_kappa"] is None


def test_wrong_route_cardinality_or_duplicate_rejected():
    ids, a, b, mapping, source = synthetic_reviews()
    with pytest.raises(ValueError):
        assess_pilot(ids[:-1],a,b,mapping,source)
    ids[0] = ids[1]
    with pytest.raises(ValueError):
        assess_pilot(ids,a,b,mapping,source)
