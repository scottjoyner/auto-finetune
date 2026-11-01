"""Synthetic only: tests for alternative LFM3 calibration direction geometry."""
import json
import pytest

from src import calibration_balance_sensitivity as sensitivity
from src import calibration_direction_stability as prior


def vector(a,b=0.):
    return [float(a),float(b)] + [0.]*(prior.WIDTH-2)


def synthetic():
    import math
    fit=vector(1/math.sqrt(5),2/math.sqrt(5))
    one=vector(1)
    two=vector(0,2)
    return {
        'format':prior.FORMAT,
        'source_model_sha256':prior.MODEL_SHA,
        'pair_dataset_sha256':prior.PAIRS_SHA,
        'direction_file_sha256':prior.DIRECTION_FILE_SHA,
        'layer':prior.LAYER,
        'hidden_size':prior.WIDTH,
        'calibration_pair_ids':list(prior.CALIBRATION),
        'holdout_pair_ids':list(prior.HOLDOUT),
        'fitted_direction':fit,
        'pair_deltas':{
            **{prior.CALIBRATION[0]:one,prior.CALIBRATION[1]:two},
            **{key:vector(1,1) for key in prior.HOLDOUT}
        },
    }


def test_balanced_is_more_aligned_on_equal_sources():
    r=sensitivity.evaluate(synthetic())
    assert r['calibration_magnitude_ratio_larger_to_smaller']==2.
    assert r['calibration_unit_cosine']==0
    assert r['calibration_pairs']==2 and r['heldout_pairs']==6
    assert list(r['candidate_diagnostics'])==list(sensitivity.CANDIDATES)
    assert r['candidate_diagnostics']['equal_unit_weight']['mean_signed_cosine'] == 1.
    assert r['candidate_diagnostics']['original_magnitude_weighted']['mean_signed_cosine'] < 1.
    assert r['balanced_minus_original_min_signed_cosine'] > 0
    assert r['independent_new_prompts']==0
    assert r['model_weights_changed'] is False
    assert r['new_model_forward_passes']==0
    assert r['human_refusal_review_complete'] is False
    assert r['model_release_approved'] is False


def test_repeatable_pinned_aggregates_do_not_expose_vectors():
    r=sensitivity.evaluate(synthetic())
    assert r==sensitivity.evaluate(synthetic())
    public=json.dumps(r)
    assert 'pair_deltas' not in public
    assert 'fitted_direction": [' not in public
    assert 'p01' not in public
    assert 'p08' not in public
    assert 'post-hoc' in r['evidence_class']


def test_anti_aligned_heldout_is_not_hidden():
    x=synthetic()
    x['pair_deltas']['p08']=vector(-1,-1)
    r=sensitivity.evaluate(x)
    for row in r['candidate_diagnostics'].values():
        assert row['positive_signed_alignment_count'] <= 5
        assert row['min_signed_cosine'] < 0


@pytest.mark.parametrize('mutate',[
    lambda x:x.update(source_model_sha256='0'*64),
    lambda x:x['pair_deltas'].pop('p07'),
    lambda x:x['fitted_direction'].__setitem__(0,0.),
    lambda x:x['pair_deltas']['p02'].__setitem__(1,0.),
    lambda x:x['pair_deltas']['p06'].__setitem__(1,float('nan')),
])
def test_rejects_invalid_or_unpinned_source(mutate):
    x=synthetic()
    mutate(x)
    with pytest.raises(ValueError):
        sensitivity.evaluate(x)


def test_very_different_calibration_magnitudes_remain_finite():
    x=synthetic()
    x['pair_deltas']['p02']=vector(0,30)
    x['fitted_direction']=prior._unit(vector(1,30))
    r=sensitivity.evaluate(x)
    assert r['calibration_magnitude_ratio_larger_to_smaller']==30.
    assert all(-1 <= entry['mean_signed_cosine'] <= 1
               for entry in r['candidate_diagnostics'].values())
    assert r['selection_was_post_hoc'] is True
