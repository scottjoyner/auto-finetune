import pytest
from src.arc_likelihood_paired_diagnostics import paired_diagnostics
from src.arc_full_likelihood_eval import ORDER


def example():
    rows=[]
    for i in range(1172):
        for arm in ORDER:
            correct=bool(i%2==0)
            if arm==ORDER[1] and i%2==0:
                correct=False
            if arm==ORDER[2]:
                correct=True
            rows.append({
                "id":f"id{i}", "arm":arm,
                "correct_index":0,
                "raw_correct":correct,
                "length_normalized_correct":correct,
                "predicted_raw_index":0 if correct else 1,
                "predicted_length_normalized_index":0 if correct else 1,
            })
    return rows


def test_paired_outcomes_are_matched():
    r=paired_diagnostics(example(),resamples=200)
    a,b=ORDER[0],ORDER[1]
    effects=r["metrics"]["length_normalized_correct"]["paired_comparisons"][f"{b}_minus_{a}"]
    assert effects["became_correct"]==0
    assert effects["became_incorrect"]==586
    assert effects["same_chosen_option"]==586
    assert effects["paired_accuracy_delta"]["mean_difference"]==-0.5
    assert r["safety_evidence"] is False


def test_duplicate_or_missing_results_rejected():
    rows=example()
    with pytest.raises(ValueError,match="full 1,172"):
        paired_diagnostics(rows[:-1],resamples=200)
    with pytest.raises(ValueError,match="duplicate"):
        paired_diagnostics(rows[:-1]+[rows[0]],resamples=200)


def test_answer_key_mismatch_rejected():
    rows=example()
    rows[0]["correct_index"]=3
    with pytest.raises(ValueError,match="answer key mismatch"):
        paired_diagnostics(rows,resamples=200)
