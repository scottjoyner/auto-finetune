"""Synthetic-only length-overlap controls; no actual XSTest answers."""
import copy
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.xstest_450_eval import ARM_ORDER
from src.xstest_length_overlap_sensitivity import analyze

A, B, C = ARM_ORDER


def panel():
    rows, cases = [], []
    for category, size, modal_count in (("safe", 250, 29), ("unsafe", 200, 127)):
        for index in range(size):
            source_id = f"sample_{category}_{index:03d}"
            cases.append({"id": source_id, "class": category, "type": "synthetic"})
            length = 10 + index % 5
            for arm in ARM_ORDER:
                if arm == C and index < modal_count:
                    response = "I cannot assist."
                elif arm == C:
                    response = "id_" + source_id + " " + " ".join(["text"] * (length - 1))
                elif arm == A:
                    response = " ".join(["alpha"] * length)
                else:
                    response = " ".join(["beta"] * (length + 2))
                rows.append({"id": source_id, "class": category, "type": "synthetic",
                             "arm": arm, "response": response,
                             "refusal_proxy": refusal_proxy(response),
                             "elapsed_seconds": 2.5 if arm == C and index < modal_count
                             else 7 + ARM_ORDER.index(arm)})
    return rows, cases


def locate(report, category, response_group, comparator, gap):
    matching = [item for item in report["results"]
                if item["prompt_class"] == category
                and item["C_output_group"] == response_group
                and item["comparator"] == comparator
                and item["max_absolute_whitespace_word_gap"] == gap]
    assert len(matching) == 1
    return matching[0]


def test_length_overlap_counts_and_no_text_leakage():
    rows, cases = panel()
    report = analyze(rows, cases)
    assert report["source_prompts"] == 450
    assert len(report["results"]) == 32
    for category, repeated, other in (("safe", 29, 221), ("unsafe", 127, 73)):
        repeated_zero = locate(report, category, "repeated", A, 0)
        other_zero = locate(report, category, "other", A, 0)
        assert repeated_zero["source_group_count"] == repeated
        assert repeated_zero["matched_length_overlap_count"] == 0
        assert repeated_zero["median_paired_C_minus_comparator_elapsed_seconds"] is None
        assert other_zero["matched_length_overlap_count"] == other
        for comparator in (A, B):
            for mode in ("repeated", "other"):
                counts = [locate(report, category, mode, comparator, gap)[
                    "matched_length_overlap_count"] for gap in (0, 5, 10, 20)]
                assert counts == sorted(counts)
                assert counts[-1] == (repeated if mode == "repeated" else other)
    serialized = json.dumps(report)
    assert "I cannot assist" not in serialized
    assert "sample_unsafe" not in serialized
    assert report["causal_speed_advantage_proven"] is False


def test_source_order_cannot_change_result():
    rows, cases = panel()
    assert analyze(rows, cases) == analyze(list(reversed(rows)), list(reversed(cases)))


@pytest.mark.parametrize("bad", [
    "missing", "duplicate", "invalid_time", "wrong_keyword",
    "changed_modal_response", "missing_source", "unknown_arm"
])
def test_tampered_inputs_fail_closed(bad):
    rows, cases = panel()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[0] = copy.deepcopy(rows[1])
    elif bad == "invalid_time":
        rows[0]["elapsed_seconds"] = float("inf")
    elif bad == "wrong_keyword":
        rows[0]["refusal_proxy"] = not rows[0]["refusal_proxy"]
    elif bad == "changed_modal_response":
        target = next(row for row in rows if row["arm"] == C and
                      row["id"] == "sample_unsafe_000")
        target["response"] = "different response"
        target["refusal_proxy"] = refusal_proxy(target["response"])
    elif bad == "missing_source":
        cases.pop()
    elif bad == "unknown_arm":
        rows[0]["arm"] = "unknown"
    with pytest.raises(ValueError):
        analyze(rows, cases)


def test_descriptive_only_and_accounting():
    rows, cases = panel()
    report = analyze(rows, cases)
    for item in report["results"]:
        assert item["descriptive_only_not_statistical_inference"]
        assert sum(item[key] for key in (
            "C_faster_count", "C_slower_count", "recorded_time_ties"
        )) == item["matched_length_overlap_count"]
        assert item["matched_length_overlap_count"] <= item["source_group_count"]
    assert not report["human_refusal_or_safety_evaluated"]
    assert not report["release_approved"]
