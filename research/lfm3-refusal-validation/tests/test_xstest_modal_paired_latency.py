"""Synthetic-only conditioned timing checks. No private answers or IDs."""
import copy
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.xstest_450_eval import ARM_ORDER
from src.xstest_modal_paired_latency import evaluate

A, B, C = ARM_ORDER


def synthetic():
    data, rows = [], []
    for group, n, repeat_n in (("safe", 250, 29), ("unsafe", 200, 127)):
        for i in range(n):
            source_id = f"{group}-{i:03d}"
            data.append({"id": source_id, "class": group, "type": "synthetic"})
            for arm in ARM_ORDER:
                repeated = arm == C and i < repeat_n
                response = "I cannot assist." if repeated else f"Distinct {group} {i} {arm}"
                elapsed = (2.5 if repeated else
                           8.0 if arm == A else
                           9.0 if arm == B else 8.5)
                rows.append({
                    "id": source_id, "class": group, "type": "synthetic",
                    "arm": arm, "response": response,
                    "refusal_proxy": refusal_proxy(response),
                    "elapsed_seconds": elapsed,
                })
    return rows, data


def by_condition(report, group, repeated, comparison):
    selected = [r for r in report["comparisons"]
                if r["prompt_class"] == group
                and r["C_response_group"] == repeated
                and r["paired_comparison"] == f"{C}_minus_{comparison}"]
    assert len(selected) == 1
    return selected[0]


def test_paired_strata_full_coverage_and_no_private_answers():
    rows, dataset = synthetic()
    report = evaluate(rows, dataset)
    assert report["source_prompts"] == 450
    assert report["modal_C_total_count"] == 156
    assert len(report["comparisons"]) == 8
    for group, repeated_n, other_n in (("safe",29,221),("unsafe",127,73)):
        for comparator in (A,B):
            repeated = by_condition(report, group, "modal_repeated", comparator)
            other = by_condition(report, group, "other", comparator)
            assert repeated["n_matched_prompts"] == repeated_n
            assert other["n_matched_prompts"] == other_n
            assert repeated["C_faster_count"] == repeated_n
            assert other["C_faster_count"] == (0 if comparator == A else other_n)
            assert sum(other[k] for k in (
                "C_faster_count","C_slower_count","recorded_time_ties")) == other_n
    assert "I cannot assist." not in json.dumps(report)
    assert "Distinct" not in json.dumps(report)
    assert "safe-001" not in json.dumps(report)
    assert report["causal_speed_advantage_proven"] is False
    assert report["human_behavioral_safety_review_complete"] is False
def test_order_independent_and_deterministic():
    rows, dataset = synthetic()
    first = evaluate(rows, dataset)
    assert first == evaluate(list(reversed(rows)), list(reversed(dataset)))


@pytest.mark.parametrize("kind", [
    "missing","duplicate","changed_mode","bad_timing","changed_flag",
    "unmatched_type","changed_benchmark",
])
def test_input_tampering_fails_closed(kind):
    rows, dataset = synthetic()
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[0] = copy.deepcopy(rows[1])
    elif kind == "changed_mode":
        target = next(r for r in rows
                      if r["arm"] == C and r["class"] == "unsafe"
                      and r["id"] == "unsafe-000")
        target["response"] = "Other text"
        target["refusal_proxy"] = refusal_proxy(target["response"])
    elif kind == "bad_timing": rows[0]["elapsed_seconds"] = float("nan")
    elif kind == "changed_flag": rows[0]["refusal_proxy"] = not rows[0]["refusal_proxy"]
    elif kind == "unmatched_type": rows[0]["type"] = "wrong"
    elif kind == "changed_benchmark": dataset.pop()
    with pytest.raises(ValueError):
        evaluate(rows, dataset)


def test_sign_test_accounting_and_multiple_test_adjustment():
    rows, dataset = synthetic()
    report = evaluate(rows, dataset)
    for item in report["comparisons"]:
        assert 0 <= item["exploratory_two_sided_sign_p"] <= 1
        assert 0 <= item["holm_adjusted_p_across_eight_posthoc_tests"] <= 1
        assert item["holm_adjusted_p_across_eight_posthoc_tests"] >= item["exploratory_two_sided_sign_p"]
