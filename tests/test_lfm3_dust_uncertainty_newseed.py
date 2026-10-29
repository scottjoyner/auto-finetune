"""Actual new-seed LFM3 frozen forecast / abstention evidence contracts."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_prequential_review import review as review_source
from src.lfm3_dust_uncertainty_replay import replay
from src.lfm3_dust_uncertainty_report import build, verify

ROOT=Path(__file__).resolve().parents[1]/"research/evidence/lfm3-dust-newseed-20261029"


def sources():
    return [json.loads((ROOT/f"seed{seed}-all-candidates-12episodes.json").read_text())
            for seed in (4,5,6)]


def summary():
    return json.loads((ROOT/"seed456-shrinkage-admission-summary.json").read_text())


@pytest.mark.parametrize("seed",[4,5,6])
def test_live_pinned_checkpoint_all_signed_candidate_receipts(seed):
    record=sources()[seed-4]
    v=review_source(record)
    assert v["status"]=="descriptive_review_required"
    assert v["prequential_cases_per_policy"]==88
    assert record["total_perturbed_forwards"]==96
    assert record["total_context_forwards"]==12
    assert record["initial_adapter_sha256"]==record["final_adapter_sha256"]
    assert record["model_adapter_update_count"]==record["total_backward_calls"]==0
    assert record["selection_partition_evaluated"] is False
    assert record["independent_test_evaluated"] is False


def test_compact_report_recomputed_identically_from_signed_direction_probes():
    source=sources()
    report=summary()
    assert verify(report,source) is True
    assert report==build(source)
    assert report["source_train_episodes"]==report["unique_training_example_keys"]==36
    assert report["status"]=="hold_not_qualified"
    assert report["weight_promotion_permitted"] is False
    assert report["optimizer_switch_permitted"] is False
    for policy in ("direction","activation","calibrated"):
        measured=report["aggregate"][policy]
        assert measured["total_scored_predictions"]==264
        assert measured["total_abstentions"]==33
        assert measured["total_recommended_probes"]==0
        assert measured["mean_seedwise_ratio_vs_zero"]>1
        assert measured["all_seed_mse_beats_zero_5pct"] is False


def test_replay_never_promotes_or_opens_locked_test():
    r=replay(sources())
    assert r["status"]=="exploratory_new_seed_prequential_replay"
    assert r["new_model_forwards_in_replay"]==0
    assert r["original_source_model_adapter_updates"]==0
    assert r["source_model_final_test_evaluated"] is False
    assert r["weight_promotion_permitted"] is False
    assert r["optimizer_switch_permitted"] is False


@pytest.mark.parametrize("patch",[
    lambda v:v[0]["episode_receipts"][3]["candidates"][0]["predictions"].update(
        {"calibrated":999}),
    lambda v:v[0]["episode_receipts"][8]["candidates"][3].update(
        {"measured_gain":42}),
    lambda v:v[1]["episode_receipts"][3].update({"calibration_event_count_after":-1}),
    lambda v:v[2]["episode_receipts"][5]["candidates"].pop(),
    lambda v:v[0].update({"model_adapter_update_count":1}),
    lambda v:v[0].update({"independent_test_evaluated":True}),
    lambda v:v[0].update({"weight_promotion_permitted":True}),
])
def test_tampered_new_run_blocks_both_replay_and_summary(patch):
    records=sources()
    patch(records)
    assert replay(records)["status"]=="blocked"
    assert build(records)["status"]=="blocked"


def test_summary_inflation_or_substituted_run_denied():
    report=summary()
    original=sources()
    tampered=copy.deepcopy(report)
    tampered["aggregate"]["calibrated"]["total_recommended_probes"]=1
    assert verify(tampered,original) is False
    tampered=copy.deepcopy(report)
    tampered["per_seed"]["6"]["policies"]["direction"]["ratio_vs_zero"]=0
    assert verify(tampered,original) is False
    replaced=copy.deepcopy(original)
    replaced[1]=copy.deepcopy(replaced[0])
    assert replay(replaced)["status"]=="blocked"


def test_registered_cold_start_not_counted_as_future_episodes():
    results=replay(sources())
    for s in ("4","5","6"):
        for head in ("direction","activation","calibrated"):
            policy=results["results"][s]["policies"][head]
            assert policy["scored_episodes"]==11
            assert policy["scored_direction_predictions"]==88
            assert policy["episode_decisions"][0]["prior_episode_count"]==0
            assert policy["episode_decisions"][1]["prior_episode_count"]==1
