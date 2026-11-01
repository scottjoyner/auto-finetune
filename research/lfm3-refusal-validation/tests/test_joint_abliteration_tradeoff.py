"""Prove joint tradeoff gate checks the original A/B/C frozen evidence."""
import copy
import json
from pathlib import Path

import pytest

from src.joint_abliteration_tradeoff import evaluate, validate_canonical

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen():
    lexical = json.loads((ROOT/"evidence/20261030_xstest_proxy_paired_exact.json").read_text())
    canonical = json.loads((ROOT/"evidence/20261030_arc_canonical_full1172_cpu_aggregate.json").read_text())
    return lexical, canonical


def test_both_benchmarks_and_all_12_comparisons(frozen):
    result=evaluate(*frozen)
    assert len(result["joint_comparisons"])==12
    assert result["new_generations"]==0
    assert result["new_checkpoint_weight_edits"]==0
    assert result["checkpoint_promotion_authorized"] is False
    assert result["human_refusal_utility_review_complete"] is False
    assert len(result["a_vs_b_canonical_capability"])==2
    assert result["benign_lexical_reduction_established"] is False


def test_canonical_a_b_effect_sizes_and_custody(frozen):
    report=evaluate(*frozen)
    rows={r["stratum_or_metric"]:r for r in report["a_vs_b_canonical_capability"]}
    raw=rows["acc"]
    normalized=rows["acc_norm"]
    assert (raw["gained_indicator"],raw["lost_indicator"])==(18,29)
    assert raw["observed_difference_pp"]==pytest.approx(-.9386)
    assert (normalized["gained_indicator"],normalized["lost_indicator"])==(14,23)
    assert normalized["observed_difference_pp"]==pytest.approx(-.7679)
    assert all(r["simultaneous_95pct_lower_pp"] < r["observed_difference_pp"] < r["simultaneous_95pct_upper_pp"] for r in rows.values())


@pytest.mark.parametrize("bad", [
    ("harness_version","0.4.12"),
    ("canonical_builtin_task_used",False),
    ("dataset_content_sha256","0"*64),
    ("n_questions_per_arm",1171),
    ("human_refusal_review_complete",True),
    ("model_release_approved",True),
])
def test_canonical_invalid_source_gates_fail(frozen,bad):
    lexical, canonical=copy.deepcopy(frozen)
    canonical[bad[0]]=bad[1]
    with pytest.raises(ValueError):
        evaluate(lexical,canonical)


@pytest.mark.parametrize("field",[
    "right_only_correct","left_only_correct",
    "right_minus_left_percentage_points","exploratory_exact_mcnemar_p",
    "holm_adjusted_p_six_comparisons",
])
def test_canonical_tampered_pair_fail_closed(frozen,field):
    lex,arc=copy.deepcopy(frozen)
    arc["matched_question_comparisons"][0][field]+=1
    with pytest.raises(ValueError):
        evaluate(lex,arc)


def test_canonical_duplicate_and_alias_fails(frozen):
    lex,arc=copy.deepcopy(frozen)
    arc["matched_question_comparisons"][-1]=arc["matched_question_comparisons"][0]
    with pytest.raises(ValueError):
        evaluate(lex,arc)
    lex,arc=copy.deepcopy(frozen)
    arc["checkpoint_sha256"]["B_M1_abliterated_smoke"]=arc["checkpoint_sha256"]["A_original_posttrained"]
    with pytest.raises(ValueError):
        evaluate(lex,arc)


def test_canonical_score_and_receipt_tamper_fail(frozen):
    lex,arc=copy.deepcopy(frozen)
    arc["metrics"]["B_M1_abliterated_smoke"]["acc"]["correct"]+=1
    with pytest.raises(ValueError):
        evaluate(lex,arc)
    lex,arc=copy.deepcopy(frozen)
    arc["private_result_and_case_sha256"]["B_M1_abliterated_smoke"]["case_sha256"]="f"*64
    with pytest.raises(ValueError):
        evaluate(lex,arc)
    lex,arc=copy.deepcopy(frozen)
    arc["canonical_vs_local_adapter_per_case"]["A_original_posttrained"]["exact_match"]=False
    with pytest.raises(ValueError):
        evaluate(lex,arc)


def test_xstest_tamper_fail_closed(frozen):
    lex,arc=copy.deepcopy(frozen)
    lex["comparisons"][0]["lexical_keyword_newly_present"]+=1
    with pytest.raises(ValueError):
        evaluate(lex,arc)
