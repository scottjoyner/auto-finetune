from copy import deepcopy
import pytest
from src.arc_scoring_sensitivity import compare_scoring
from src.arc_full_likelihood_eval import ORDER


def dataset():
    cases=[]
    for case in range(1172):
        for arm in ORDER:
            cases.append({
                "arm":arm,"id":str(case),
                "candidate_token_lengths":[2,3,4,5],
                "predicted_raw_index":0,
                "predicted_length_normalized_index":1,
                "raw_correct":False,
                "length_normalized_correct":True,
            })
    return cases


def test_sensitivity_is_privacy_safe_aggregate_only():
    out=compare_scoring(dataset())
    assert out["questions"]==1172
    assert out["uniform_random_option_expected_accuracy"]==0.25
    assert out["unique_shortest_option_cases"]==1172
    assert out["by_arm"][ORDER[0]]["normalized_rescued_correct"]==1172
    assert out["by_arm"][ORDER[0]]["unique_shortest_selected_raw"]==1172
    assert out["by_arm"][ORDER[0]]["unique_shortest_selected_normalized"]==0
    assert out["safety_validated"] is False


def test_reject_mismatched_option_tokenization():
    rows=dataset()
    rows[1]["candidate_token_lengths"]=[2,2,4,5]
    with pytest.raises(ValueError,match="tokenization"):
        compare_scoring(rows)


def test_reject_duplicate_and_incomplete():
    rows=dataset()
    with pytest.raises(ValueError,match="requires 1,172"):
        compare_scoring(rows[:-1])
    with pytest.raises(ValueError,match="duplicate"):
        compare_scoring(rows[:-1]+[deepcopy(rows[0])])
