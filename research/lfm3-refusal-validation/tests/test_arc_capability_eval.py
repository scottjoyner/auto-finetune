import pytest
from src.arc_capability_eval import normalize_case, candidate_ids


def row():
    return {
        "id":"sample",
        "question":"Pick the correct letter.",
        "choices":{"label":["1","2","3"],"text":["first","second","third"]},
        "answerKey":"2"
    }


def test_normalize_numeric_labels():
    case=normalize_case(row())
    assert case["correct"]=="B"
    assert "A. first" in case["prompt"]
    assert "C. third" in case["prompt"]


def test_answer_not_present_rejected():
    r=row();r["answerKey"]="99"
    with pytest.raises(ValueError,match="answerKey"):
        normalize_case(r)


def test_multitoken_answer_rejected():
    class FakeTokenizer:
        def encode(self,txt,add_special_tokens=False):return [3,4]
    with pytest.raises(ValueError,match="single-token"):
        candidate_ids(FakeTokenizer(),"AB")
