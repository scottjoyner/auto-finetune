"""Contract for the actual, aggregate-only independent-corpus geometry evidence."""
import json
from pathlib import Path

from src import xstest_direction_transfer as study
from src import calibration_direction_stability as frozen


EVIDENCE = (Path(__file__).resolve().parents[1] / 'evidence' /
            '20261101_xstest_direction_transfer_54_actual.json')


def test_actual_evidence_is_hash_pinned_and_private_free():
    raw=EVIDENCE.read_text()
    r=json.loads(raw)
    assert r['source_model_sha256']==frozen.MODEL_SHA
    assert r['source_direction_sha256']==frozen.DIRECTION_FILE_SHA
    assert r['dataset_sha256']==study.SOURCE_SHA256
    assert r['prompt_count']==54
    assert r['safe_prompts']==30
    assert r['unsafe_prompts']==24
    assert r['category_count']==18
    assert r['isotropic_sham_seed_first']==31000
    assert r['isotropic_sham_count']==128
    assert r['isotropic_shams_are_null_distribution'] is False
    assert r['new_model_forward_passes']==54
    assert r['model_generations']==0
    assert r['model_weights_changed'] is False
    assert r['human_refusal_grades']==0
    assert r['model_release_approved'] is False
    assert r['raw_prompts_or_activations_exported'] is False
    forbidden=('pair_deltas','fitted_direction": [','review_id',
               'selection_ids','source_id','raw_response','raw_prompt',
               '"activation":', '"prompt":', '"response":')
    assert not any(token in raw for token in forbidden)


def test_reported_candidate_directions_are_distinct_and_not_promoted():
    r=json.loads(EVIDENCE.read_text())
    results=r['candidate_class_cosine']
    assert set(results)=={
        'original_M1_fitted','equal_unit_weight',
        'calibrator_1_only','calibrator_2_only',
    }
    assert results['original_M1_fitted']['unsafe_minus_safe_cosine']==.144124
    assert results['equal_unit_weight']['unsafe_minus_safe_cosine']==.12559
    assert results['calibrator_1_only']['unsafe_minus_safe_cosine']==.032564
    assert results['calibrator_2_only']['unsafe_minus_safe_cosine']==.153417
    assert r['isotropic_sham_abs_class_gap_max']==.02001
    assert r['isotropic_shams_with_abs_gap_at_least_fitted']==0
    assert r['candidate_selection_posthoc'] is True
    assert 'not refusal' in r['evidence_class']
    assert 'not refusal prediction' in ' '.join(r['limitations'])
