"""Synthetic-only checks. Do not label these outputs empirical LFM3 evidence."""
import json
from pathlib import Path
import stat

import pytest

from src import calibration_direction_stability as stable


def unit(a, b=0.):
    return [a, b] + [0.]*(stable.WIDTH-2)


def fixture():
    one = unit(1)
    return {
        'format': stable.FORMAT,
        'source_model_sha256': stable.MODEL_SHA,
        'pair_dataset_sha256': stable.PAIRS_SHA,
        'direction_file_sha256': stable.DIRECTION_FILE_SHA,
        'layer': 10,
        'hidden_size': stable.WIDTH,
        'calibration_pair_ids': list(stable.CALIBRATION),
        'holdout_pair_ids': list(stable.HOLDOUT),
        'fitted_direction': one,
        'pair_deltas': {p: list(one) for p in stable.ALL_IDS},
    }


def test_identical_directions_and_all_shams_deterministic():
    input_ = fixture()
    result = stable.audit(input_)
    assert result == stable.audit(input_)
    assert result['calibration_leave_one_out_cosine'] == 1.
    assert result['heldout_fitted_alignment_mean'] == 1.
    assert result['heldout_positive_direction_matches'] == 6
    assert result['frozen_fitted_direction_reconstruction_cosine'] == 1.
    assert len(result['independent_sham_directions_orthogonal_to_fit']) == 4
    assert all(row['fitted_higher_absolute_alignment_cases'] == 6 for row in result['independent_sham_directions_orthogonal_to_fit'])
    assert result['model_release_approved'] is False
    assert result['model_weights_changed'] is False
    assert result['claim_of_direction_stability_verified'] is False
    text = json.dumps(result)
    assert 'pair_deltas' not in text and 'fitted_direction": [' not in text
    assert 'p01' not in text and 'p08' not in text


def test_synthetic_anti_aligned_holdouts_are_measured_not_hidden():
    input_ = fixture()
    for p in stable.HOLDOUT[:3]:
        input_['pair_deltas'][p] = unit(-1)
    res = stable.audit(input_)
    assert res['heldout_positive_direction_matches'] == 3
    assert res['heldout_fitted_alignment_min'] == -1
    assert res['heldout_fitted_alignment_mean'] == 0
    assert res['heldout_fitted_absolute_alignment_mean'] == 1


def test_calibration_direction_fails_when_fitted_is_different():
    input_ = fixture()
    input_['fitted_direction'] = unit(0, 1)
    with pytest.raises(ValueError, match='reproduce'):
        stable.audit(input_)


def test_pairwise_calibration_mismatch_not_masked_by_reference():
    input_ = fixture()
    input_['pair_deltas']['p02'] = unit(0,1)
    x = 2**-.5
    input_['fitted_direction'] = unit(x,x)
    r = stable.audit(input_)
    assert r['calibration_leave_one_out_cosine'] == 0
    assert r['heldout_fitted_alignment_mean'] == pytest.approx(x,abs=1e-6)


def test_cancelled_calibration_detected():
    input_ = fixture()
    input_['pair_deltas']['p02'] = unit(-1)
    with pytest.raises(ValueError, match='cancelling'):
        stable.audit(input_)


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(source_model_sha256='0'*64),
    lambda x: x.update(pair_dataset_sha256='0'*64),
    lambda x: x.update(direction_file_sha256='0'*64),
    lambda x: x.update(layer=11),
    lambda x: x.update(hidden_size=1024),
    lambda x: x.update(holdout_pair_ids=list(reversed(x['holdout_pair_ids']))),
    lambda x: x.update(calibration_pair_ids=['p01','p01']),
    lambda x: x['pair_deltas'].pop('p08'),
    lambda x: x['pair_deltas'].update(p09=unit(1)),
    lambda x: x.update(extra_private_prompt='please do something'),
    lambda x: x['pair_deltas']['p08'].__setitem__(0,float('nan')),
    lambda x: x['pair_deltas']['p08'].__setitem__(0,True),
    lambda x: x['pair_deltas']['p08'].__setitem__(0,0.),
    lambda x: x['fitted_direction'].__setitem__(0,float('inf')),
    lambda x: x.update(layer=True),
])
def test_malformed_or_spoofed_source_rejected(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError):
        stable.audit(data)


def test_private_source_and_output_permissions(tmp_path):
    input_ = fixture()
    safe = tmp_path/'private'
    safe.mkdir(mode=0o700)
    vec = safe/'vector.json'
    vec.write_text(json.dumps(input_))
    vec.chmod(0o600)
    assert stable.audit(stable._private_input(vec))['calibration_pairs'] == 2
    dst=safe/'out.json'
    stable.write_exclusive(dst, stable.audit(input_))
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        stable.write_exclusive(dst, {'should_not_overwrite':True})
    vec.chmod(0o644)
    with pytest.raises(PermissionError):
        stable._private_input(vec)
    vec.chmod(0o600)
    link=safe/'link.json'
    link.symlink_to(vec)
    with pytest.raises(ValueError,match='symlink'):
        stable._private_input(link)
    safe.chmod(0o755)
    with pytest.raises(PermissionError):
        stable._private_input(vec)
