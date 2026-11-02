"""Synthetic-only timing tests. Private responses never leave the node."""
import copy
import json
import math
import pytest

from src.eval_contracts import refusal_proxy
from src.xstest_450_eval import ARM_ORDER
from src.xstest_latency_repetition import summarize


def fixture_panel():
    data,rows=[],[]
    for group,n in (("safe",250),("unsafe",200)):
        for i in range(n):
            case=f"{group}_{i:03d}"
            data.append({"id":case,"class":group,"type":"synthetic"})
            for j,arm in enumerate(ARM_ORDER):
                common=j==2 and i<(29 if group=="safe" else 127)
                reply="I cannot assist." if common else f"Unique {group} {i} arm {j}"
                elapsed=(2.5 if common else 8.5+j+(i%3)*.5)
                rows.append({
                    "id":case,"class":group,"type":"synthetic",
                    "arm":arm,"response":reply,"refusal_proxy":refusal_proxy(reply),
                    "elapsed_seconds":elapsed,
                })
    return rows,data


def test_full_journal_latency_and_mode_aggregate():
    rows,data=fixture_panel()
    out=summarize(rows,data)
    assert out["n_generations"]==1350
    assert out["per_stratum"]["safe"]["arms"][ARM_ORDER[2]]["short_modal_response"]["repeat_count"]==29
    assert out["per_stratum"]["unsafe"]["arms"][ARM_ORDER[2]]["short_modal_response"]["repeat_count"]==127
    assert out["per_stratum"]["unsafe"]["arms"][ARM_ORDER[2]]["short_modal_response"]["median_repeated_elapsed_seconds"]==2.5
    assert len(out["paired_elapsed_diagnostics"])==6
    assert not out["decode_tokens_per_second_measured"]
    assert not out["actual_human_safety_validated"]
    assert "Unique unsafe" not in json.dumps(out)
@pytest.mark.parametrize("bad",[
    "missing","duplicate","wrong_class","wrong_type","unknown_arm","unknown_id",
    "empty_response","bad_keyword","zero_time","negative_time","nan_time","infinite_time","bool_time",
])
def test_bad_journal_fails_closed(bad):
    rows,data=fixture_panel()
    if bad=="missing": rows.pop()
    elif bad=="duplicate": rows[0]=copy.deepcopy(rows[1])
    elif bad=="wrong_class": rows[0]["class"]="unsafe"
    elif bad=="wrong_type": rows[0]["type"]="other"
    elif bad=="unknown_arm": rows[0]["arm"]="D"
    elif bad=="unknown_id": rows[0]["id"]="new_case"
    elif bad=="empty_response": rows[0]["response"]=None
    elif bad=="bad_keyword": rows[0]["refusal_proxy"]=True
    elif bad=="zero_time": rows[0]["elapsed_seconds"]=0
    elif bad=="negative_time": rows[0]["elapsed_seconds"]=-1
    elif bad=="nan_time": rows[0]["elapsed_seconds"]=float("nan")
    elif bad=="infinite_time": rows[0]["elapsed_seconds"]=float("inf")
    elif bad=="bool_time": rows[0]["elapsed_seconds"]=True
    with pytest.raises(ValueError):
        summarize(rows,data)


def test_repeated_C_count_and_mode_keyword_pinned():
    rows,data=fixture_panel()
    for row in rows:
        if row["arm"]==ARM_ORDER[2] and row["class"]=="safe" and row["id"]=="safe_000":
            row["response"]="Different answer";row["refusal_proxy"]=refusal_proxy(row["response"])
            break
    with pytest.raises(ValueError,match="expected single frozen"):
        summarize(rows,data)


def test_elapsed_sign_inference_is_multiplicity_adjusted():
    rows,data=fixture_panel()
    report=summarize(rows,data)
    for comparison in report["paired_elapsed_diagnostics"]:
        assert sum(comparison[k] for k in ("right_slower","right_faster","elapsed_time_ties"))==comparison["matched_prompts"]
        assert 0<=comparison["holm_adjusted_p_for_six_exploratory_tests"]<=1
