"""Causal OOD threshold must use TRAIN summaries only, no transfer gains."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_task_shift_probe import NEW_SEEDS,curriculum,directions,task_hash

ROOT=Path(__file__).resolve().parents[1]/"research/evidence/lfm3-dust-text-transfer-20261029"


def source(seed):
    return json.loads((ROOT/f"seed{seed}-extraction-to-uppercase-18episodes.json").read_text())


@pytest.mark.parametrize("seed",[10,11,12])
def test_train_only_loo_threshold_accepts_all_training_contexts_and_detects_uppercase_shift(seed):
    raw=source(seed)
    gate=FrozenTaskShiftGate(source=raw)
    assert gate.train_feedback_sample_count==80
    assert gate.calibration_sample_count==12
    assert gate.training_loo_min_nearest-gate.threshold==pytest.approx(.02)
    assert gate.verified_skill is False
    assert gate.summary()["source_transfer_episodes_used_to_fit_gate"]==0
    for index,event in enumerate(raw["events"][:12]):
        signal=gate.evaluate(event["context"],1e-4,candidate_index=0)
        assert signal["in_training_context_envelope"] is True
        assert signal["selected_for_live_probe"] is False
    for event in raw["events"][12:]:
        signal=gate.evaluate(event["context"],1e-4,candidate_index=0)
        assert signal["in_training_context_envelope"] is False
        assert signal["abstention_reason"]=="novel_activation_context"
        assert signal["guarded_head_prediction"]==0
        assert signal["candidate_measured_gain"] is None
        assert signal["weight_promotion_permitted"] is False


@pytest.mark.parametrize("seed",[10,11,12])
def test_guard_does_not_consult_existing_transfer_labels(seed):
    raw=source(seed)
    original=FrozenTaskShiftGate(source=raw)
    altered=copy.deepcopy(raw)
    for row in altered["events"][12:]:
        row["context"]=[1.,0.,0.,0.]
        for candidate in row["signed_directions"]:
            candidate["measured_gain"]=100.
            candidate["predictions"]["neural"]=-100.
    altered_gate=FrozenTaskShiftGate(source=altered)
    assert altered_gate.summary()==original.summary()


def test_new_novel_task_seeds_and_prompts_are_strictly_unique():
    keys=[]
    for seed in NEW_SEEDS:
        rows=curriculum(seed)
        assert len(rows)==6
        assert all(x["family"]=="sort_two_words" for x in rows)
        keys.extend(r["input_key"] for r in rows)
        assert all("Sort these two lowercase words" in x["prompt"] for x in rows)
        assert len({task_hash(x) for x in rows})==6
        for ep in range(6):
            d=directions(seed,ep)
            assert len(d)==8
            for i in range(0,8,2):
                assert d[i][0]==d[i+1][0]
                assert d[i][1]==1 and d[i+1][1]==-1
                assert all(a==-b for a,b in zip(d[i][2],d[i+1][2]))
    assert len(set(keys))==18
    with pytest.raises(ValueError):
        curriculum(16)


def test_malformed_attention_context_and_candidate_cannot_authorize():
    gate=FrozenTaskShiftGate(source=source(10))
    for ctx in ([0.,0.,0.,0.],[1.,0.,0.], [float("nan"),0.,0.,0.]):
        with pytest.raises(ValueError):
            gate.evaluate(ctx,0.,candidate_index=0)
    with pytest.raises(ValueError):
        gate.evaluate([1.,0.,0.,0.],float("nan"),candidate_index=1)
    with pytest.raises(ValueError):
        gate.evaluate([1.,0.,0.,0.],.1,candidate_index=8)
