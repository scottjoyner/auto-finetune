from types import SimpleNamespace
import pytest
import torch

from src.arc_full_likelihood_eval import prepare,score_case,resume,manifest_for,ORDER


class Tokenizer:
    pad_token_id=0
    eos_token_id=0
    def encode(self,txt,add_special_tokens=False):
        return [(ord(c)%50)+1 for c in txt]


class DummyModel:
    def __call__(self,input_ids,attention_mask,use_cache,logits_to_keep):
        b,n=input_ids.shape
        scores=torch.zeros((b,n,80))
        # Favor token 'a' more than others.
        return SimpleNamespace(logits=scores)


def row():
    return {"id":"q1","question":"Which one?","choices":{
        "label":["1","2"],"text":["Alpha","Beta"]},"answerKey":"1"}


def test_canonical_choice_mapping():
    case=prepare(row())
    assert case["correct_index"]==0
    assert case["options"]==["Alpha","Beta"]


def test_choice_text_scoring_is_not_single_letter_proxy():
    out=score_case(DummyModel(),Tokenizer(),prepare(row()))
    assert out["id"]=="q1"
    assert len(out["mean_per_token_logprobs"])==2
    assert out["candidate_token_lengths"]==[len(" Alpha"),len(" Beta")]
    assert all(v<0 for v in out["raw_logprobs"])


def test_resume_three_arm_order():
    cases=[{"id":"q1"},{"id":"q2"}]
    seen=resume([{"arm":ORDER[0],"id":"q1"},{"arm":ORDER[0],"id":"q2"}],cases)
    assert len(seen)==2


def test_duplicate_or_out_of_order_rejected():
    cases=[{"id":"q1"},{"id":"q2"}]
    with pytest.raises(ValueError,match="duplicate"):
        resume([{"arm":ORDER[0],"id":"q1"}]*2,cases)
    with pytest.raises(ValueError,match="ordering"):
        resume([{"arm":ORDER[0],"id":"q2"},{"arm":ORDER[0],"id":"q1"}],cases)


def test_bad_dataset_case():
    wrong=row()
    wrong["answerKey"]="other"
    with pytest.raises(ValueError,match="invalid question"):
        prepare(wrong)
