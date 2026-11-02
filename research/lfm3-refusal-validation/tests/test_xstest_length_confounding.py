"""Synthetic-only validation; none of the private 1,350 responses are read."""
import copy
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.xstest_450_eval import ARM_ORDER
from src.xstest_length_confounding import percentile, summarize


def synthetic_panel():
    cases = []
    rows = []
    for cls, count in (("safe", 250), ("unsafe", 200)):
        for i in range(count):
            case = f"{cls}{i:03d}"
            source = {"id": case, "class": cls, "type": "synthetic_" + cls}
            cases.append(source)
            for j, arm in enumerate(ARM_ORDER):
                response = ["I cannot assist.", "I can provide detailed help.", "No."][j]
                rows.append({"id": case, "class": cls, "type": source["type"],
                             "arm": arm, "response": response,
                             "refusal_proxy": refusal_proxy(response)})
    return rows, cases
def test_all_450_by_three_aggregates_and_invariants():
    rows, cases = synthetic_panel()
    result = summarize(rows, cases)
    assert result["n_generations"] == 1350
    assert result["strata"]["unsafe"]["per_arm"][ARM_ORDER[2]]["short_response_less_than_50_characters"] == 200
    assert result["strata"]["safe"]["per_arm"][ARM_ORDER[0]]["short_response_keyword_positive"] == 250
    assert result["human_behavioral_refusals_verified"] is False
    assert result["causal_length_effect_established"] is False
    assert len(result["paired_word_length_diagnostics"]) == 6
    assert "safe000" not in json.dumps(result)
    assert "I cannot assist." not in json.dumps(result)


@pytest.mark.parametrize("tamper", ["missing", "duplicated", "type", "class",
                                    "keyword", "response", "arm", "case_id"])
def test_fail_closed_on_record_tampering(tamper):
    rows, cases = synthetic_panel()
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicated":
        rows[0] = copy.deepcopy(rows[1])
    elif tamper == "type":
        rows[0]["type"] = "other_type"
    elif tamper == "class":
        rows[0]["class"] = "unsafe"
    elif tamper == "keyword":
        rows[0]["refusal_proxy"] = False
    elif tamper == "response":
        rows[0]["response"] = None
    elif tamper == "arm":
        rows[0]["arm"] = "D_unapproved"
    elif tamper == "case_id":
        rows[0]["id"] = "unknown_case"
    with pytest.raises(ValueError):
        summarize(rows, cases)
def test_percentile_and_short_threshold():
    assert percentile([5, 2, 8, 1], .1) == 1
    assert percentile([5, 2, 8, 1], .5) == 2
    assert percentile([5, 2, 8, 1], .9) == 8
    with pytest.raises(ValueError):
        percentile([], .5)


def test_short_label_is_strict_less_than_50():
    rows, cases = synthetic_panel()
    for row in rows:
        if row["class"] == "unsafe" and row["arm"] == ARM_ORDER[2]:
            row["response"] = "x" * 50
            row["refusal_proxy"] = False
    out = summarize(rows, cases)
    c = out["strata"]["unsafe"]["per_arm"][ARM_ORDER[2]]
    assert c["short_response_less_than_50_characters"] == 0
    assert c["long_response_at_least_50_characters"] == 200
