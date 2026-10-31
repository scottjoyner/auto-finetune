"""Synthetic only, without access to private model generations."""
import copy
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.xstest_450_eval import ARM_ORDER
from src.xstest_response_diversity import summarize


def fake_panel():
    cases, records = [], []
    for group, n in (("safe", 250), ("unsafe", 200)):
        for i in range(n):
            ident = f"{group}_{i:03d}"
            case = {"id": ident, "class": group, "type": "synthetic_" + group}
            cases.append(case)
            for j, arm in enumerate(ARM_ORDER):
                response = f"Ordinary distinct response {group} {i} {j}."
                if j == 2 and i < (29 if group == "safe" else 127):
                    response = "I cannot assist."
                records.append({
                    "id": ident, "class": group, "type": case["type"],
                    "arm": arm, "response": response,
                    "refusal_proxy": refusal_proxy(response),
                })
    return records, cases


def test_counts_and_cross_stratum_repeated_mode():
    rows, dataset = fake_panel()
    out = summarize(rows, dataset)
    c = out["checkpoint_data"][ARM_ORDER[2]]
    assert out["n_generations"] == 1350
    assert c["overall"]["distinct_exact_responses"] == 295
    assert c["overall"]["largest_identical_response_group"] == 156
    assert c["overall"]["top_mode"]["safe_count"] == 29
    assert c["overall"]["top_mode"]["unsafe_count"] == 127
    assert c["overall"]["responses_matching_cross_stratum_strings"] == 156
    assert c["strata"]["unsafe"]["distinct_keyword_positive_responses"] == 1
    assert out["human_behavioral_refusal_verified"] is False
    assert "I cannot assist." not in json.dumps(out)
    assert "safe_000" not in json.dumps(out)
def test_all_unique_mode_is_unset_to_avoid_tie_breaking():
    rows, dataset = fake_panel()
    report = summarize(rows, dataset)
    for arm in ARM_ORDER[:2]:
        a = report["checkpoint_data"][arm]["overall"]
        assert a["distinct_exact_responses"] == 450
        assert a["largest_identical_response_group"] == 1
        assert a["most_common_group_is_unique"] is False
        assert a["top_mode"] is None


@pytest.mark.parametrize("kind", [
    "missing", "duplicate", "wrong_flag", "wrong_group",
    "wrong_type", "invalid_response", "unknown_arm", "unknown_case",
])
def test_fail_closed_on_untrusted_panel(kind):
    rows, dataset = fake_panel()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[0] = copy.deepcopy(rows[1])
    elif kind == "wrong_flag":
        rows[0]["refusal_proxy"] = True
    elif kind == "wrong_group":
        rows[0]["class"] = "unsafe"
    elif kind == "wrong_type":
        rows[0]["type"] = "other"
    elif kind == "invalid_response":
        rows[0]["response"] = None
    elif kind == "unknown_arm":
        rows[0]["arm"] = "D"
    elif kind == "unknown_case":
        rows[0]["id"] = "missing"
    with pytest.raises(ValueError):
        summarize(rows, dataset)
def test_reject_duplicate_source_case():
    rows, dataset = fake_panel()
    dataset[0]["id"] = dataset[1]["id"]
    with pytest.raises(ValueError):
        summarize(rows, dataset)


def test_reject_bad_class_distribution():
    rows, dataset = fake_panel()
    dataset[0]["class"] = "unsafe"
    with pytest.raises(ValueError):
        summarize(rows, dataset)
