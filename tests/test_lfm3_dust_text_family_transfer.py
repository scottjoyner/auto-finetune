"""Contracts for new task-family-disjoint LFM3 trainable Dust experiment."""
import math
import pytest
from src.lfm3_dust_text_family_curriculum import (
    SEEDS,TRAIN_EPISODES,TRANSFER_EPISODES,curriculum,manifest,
)
from src.lfm3_dust_text_family_transfer_probe import validate,score_transfer,POLICIES
from src.lfm3_dust_trainable_attention import ForwardOnlyNeuralAttentionHead


def test_public_dataset_separate_families_and_no_reused_examples():
    ids=[]
    for s in SEEDS:
        data=curriculum(s)
        assert len(data["train"])==TRAIN_EPISODES==12
        assert len(data["transfer_evaluation"])==TRANSFER_EPISODES==6
        assert {x["family"] for x in data["train"]}=={"first_word","last_word"}
        assert {x["family"] for x in data["transfer_evaluation"]}=={"uppercase"}
        assert len({x["input_key"] for x in sum(data.values(),[])})==18
        for r in sum(data.values(),[]):
            assert r["answer"]!=r["foil"]
            assert all(ord(c)<128 for c in r["prompt"]+r["answer"])
        assert len(manifest(s)["train"])==64
        ids += [r["input_key"] for r in sum(data.values(),[])]
    # Across seeds: never reuse identical instruction+operand example.
    assert len(set(ids))==54


def test_only_preregistered_12_plus_6_trial_scope():
    for seed in SEEDS:
        validate(seed=seed)
        assert len(ForwardOnlyNeuralAttentionHead(seed=seed).params)==195
    for trial in (
        {"seed":9}, {"seed":13},{"seed":10,"train_episodes":13},
        {"seed":11,"transfer_episodes":5},
        {"seed":12,"signed_directions":4},
        {"seed":10,"sigma":0.1},
    ):
        with pytest.raises(ValueError):
            validate(**trial)


def test_transfer_score_exactly_six_episodes_and_all_signed_directions():
    events=[]
    for ep in range(12,18):
        rows=[{"measured_gain":float(i-3)/100,
               "predictions":{name:float(i-3)/100 if name=="neural" else 0.
                              for name in POLICIES}}
              for i in range(8)]
        events.append({"episode":ep,"phase":"transfer_evaluation","signed_directions":rows})
    stats=score_transfer(events)
    assert stats["neural"]["mean_squared_error"]==pytest.approx(0.)
    assert stats["neural"]["prospective_cases"]==48
    assert stats["zero"]["mean_squared_error"]>0.
    with pytest.raises(ValueError):
        score_transfer(events[:-1])
    with pytest.raises(ValueError):
        score_transfer(events[:1]+[{"episode":13,"phase":"transfer_evaluation",
                                    "signed_directions":events[1]["signed_directions"][:7]}]+events[2:])


def test_cold_neural_head_can_forecast_without_test_feedback():
    h=ForwardOnlyNeuralAttentionHead(seed=10)
    d=(1.,0.,0.,0.,0.,0.,0.,0.)
    c=(1.,0.,0.,0.)
    assert h.predict(d,c,episode=12)["historical_keys"]==0
    assert h.parameter_sha256==ForwardOnlyNeuralAttentionHead(seed=10).parameter_sha256
