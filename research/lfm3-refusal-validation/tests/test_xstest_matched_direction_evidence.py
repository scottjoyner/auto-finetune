"""Immutable public-safe contract for measured, matched LFM3 XSTest geometry."""
import json
from pathlib import Path

from src import xstest_matched_direction_transfer as study
from src import calibration_direction_stability as frozen
from src.xstest_450_eval import SOURCE_SHA256

FILE=Path(__file__).resolve().parents[1]/'evidence'/'20261101_xstest_matched_18_actual.json'


def test_actual_evidence_pins_source_and_contains_only_aggregate():
    s=FILE.read_text()
    x=json.loads(s)
    assert x['source_model_sha256']==frozen.MODEL_SHA
    assert x['source_direction_sha256']==frozen.DIRECTION_FILE_SHA
    assert x['source_dataset_sha256']==SOURCE_SHA256
    assert x['matched_pairs']==18
    assert x['source_category_families']==6
    assert x['new_model_forward_passes']==36
    assert x['layer']==10
    assert x['isotropic_sham_count']==128
    assert x['shams_are_causal_null'] is False
    assert x['matched_family_choice_posthoc_to_prior_xstest_activation_result'] is True
    assert x['models_weights_changed'] is False
    assert x['model_generations']==0
    assert x['human_refusal_grades']==0
    assert x['model_release_approved'] is False
    assert x['raw_prompts_or_pair_activations_exported'] is False
    forbidden=('\"safe_activation\"','\"unsafe_activation\"',
               '\"prompt\"','\"response\"','\"case_id\"',
               '\"source_id\"','\"reviewer_id\"','\"pair_deltas\"')
    assert not any(token in s for token in forbidden)


def test_actual_matched_metric_consistency_and_no_behavioral_claim():
    x=json.loads(FILE.read_text())
    by=x['candidate_matched_diagnostics']
    assert set(by)=={'original_M1_fitted','equal_unit_weight',
                    'calibrator_1_only','calibrator_2_only'}
    assert by['original_M1_fitted']['mean_signed_unsafe_minus_safe_cosine']==.17959
    assert by['original_M1_fitted']['median_signed_pair_delta']==.163781
    assert by['original_M1_fitted']['positive_matched_pair_count']==18
    assert by['original_M1_fitted']['positive_family_mean_count']==6
    assert by['equal_unit_weight']['mean_signed_unsafe_minus_safe_cosine']==.154175
    assert by['calibrator_1_only']['mean_signed_unsafe_minus_safe_cosine']==.032841
    assert by['calibrator_2_only']['mean_signed_unsafe_minus_safe_cosine']==.19547
    assert x['isotropic_sham_maximum_absolute_mean_gap']==.025893
    assert x['isotropic_shams_at_least_as_large_as_fitted']==0
    assert 'not behavioral refusal' in x['evidence_class']
