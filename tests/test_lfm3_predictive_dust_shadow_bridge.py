"""Live-source head reconstruction must remain shadow-only after admission HOLD."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_text_transfer_admission import assess
from src.lfm3_predictive_dust_shadow_bridge import replay_frozen_neural_shadow

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"research/evidence/lfm3-dust-text-transfer-20261029"


def receipts():
    return [json.loads((SOURCE/f"seed{s}-extraction-to-uppercase-18episodes.json").read_text())
            for s in (10,11,12)]


@pytest.mark.parametrize("seed", (10,11,12))
def test_trained_195_weight_model_head_uses_real_frozen_forecast_with_no_actuation(seed):
    all_reports=receipts()
    admission=assess(all_reports)
    result=replay_frozen_neural_shadow(all_reports[seed-10],admission)
    assert result["status"]=="predictive_head_shadow_abstention"
    assert result["seed"]==seed
    assert result["head_policy_qualified"] is False
    assert result["actual_head_parameter_count"]==195
    assert result["unseen_task_family"]=="uppercase"
    assert result["evaluated_using_preprobe_forecasts_only"] is True
    assert result["actual_adapter_updates_performed"]==0
    assert result["additional_model_forwards_performed"]==0
    assert result["selected_for_live_probe"] is None
    assert result["model_training_permitted"] is False
    assert result["optimizer_switch_permitted"] is False
    assert result["weight_promotion_permitted"] is False
    assert len(result["candidate_order_high_to_low"])==8
    assert sorted(result["candidate_order_high_to_low"])==list(range(8))


def test_nonfirst_unseen_prompt_still_frozen_and_never_trains():
    reports=receipts()
    a=assess(reports)
    for index in (1,5):
        result=replay_frozen_neural_shadow(reports[0],a,transfer_index=index)
        assert result["status"]=="predictive_head_shadow_abstention"
        assert result["episode"]==12+index
        assert result["selected_for_live_probe"] is None
        assert result["actual_adapter_updates_performed"]==0


@pytest.mark.parametrize("edit",[
    lambda report:report["events"][12]["signed_directions"][0]["predictions"].update({"neural":10}),
    lambda report:report["events"][9]["signed_directions"][0].update({"measured_gain":88}),
    lambda report:report["events"][14].update({"neural_head_sha_after_episode":"e"*64}),
    lambda report:report.update({"head_frozen_before_transfer_sha256":"1"*64}),
])
def test_corrupt_model_head_or_transfer_labels_fail_closed(edit):
    reports=receipts()
    a=assess(reports)
    forged=copy.deepcopy(reports[0])
    edit(forged)
    assert replay_frozen_neural_shadow(forged,a)["status"]=="blocked"


def test_favorable_or_missing_admission_cannot_activate_head():
    reports=receipts()
    a=assess(reports)
    a["status"]="separate_confirmation_study_only"
    assert replay_frozen_neural_shadow(reports[0],a)["status"]=="blocked"
    a["status"]="hold_not_qualified"
    a["predictor_qualified_for_model_updates"]=True
    assert replay_frozen_neural_shadow(reports[0],a)["status"]=="blocked"
    assert replay_frozen_neural_shadow(reports[0],{},transfer_index=0)["status"]=="blocked"
    a["predictor_qualified_for_model_updates"]=False
    assert replay_frozen_neural_shadow(reports[0],a,transfer_index=6)["status"]=="blocked"
