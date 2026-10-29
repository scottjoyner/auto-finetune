"""Require predictor superiority over naive controls before new trials."""
import copy
import json
from pathlib import Path

from src.lfm3_dust_predictor_admission import assess

ROOT=Path(__file__).resolve().parents[1]/"research/evidence/lfm3-dust-prequential-20261029"


def records():
    return [json.loads((ROOT/f"seed{s}-all-candidates-8episodes.json").read_text())
            for s in (1,2,3)]


def test_live_evidence_explicitly_vetoes_auto_adaptation():
    result=assess(records())
    assert result["status"]=="hold_not_qualified"
    assert result["unique_training_prompt_episodes"]==24
    assert "insufficient_independent_source_episodes_24_lt_32" in result["blocking_reasons"]
    for name in ("direction","activation","calibrated"):
        a=result["candidate_results"][name]
        assert a["mse_gate"] is False
        assert a["eligible_for_independent_confirmation"] is False
    assert result["optimizer_switch_permitted"] is False
    assert result["model_update_permitted"] is False
    assert result["weight_promotion_permitted"] is False


def test_duplicate_or_tampered_report_denied():
    data=records()
    assert assess([data[0],data[0],data[2]])["status"]=="blocked"
    changed=copy.deepcopy(data)
    changed[0]["metrics"]["activation"]["mse"]=0
    assert assess(changed)["status"]=="blocked"


def test_missing_report_denied():
    assert assess(records()[:2])["status"]=="blocked"
    assert assess(records(),minimum_independent_episodes=1)["status"]=="blocked"


def test_renaming_seed_does_not_prove_independent_run():
    data=records()
    data[2]["seed"]=1
    assert assess(data)["status"]=="blocked"
