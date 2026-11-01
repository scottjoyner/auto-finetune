"""Synthetic-only source contract tests for matched XSTest direction geometry."""
import math
import copy
import pytest

from src import xstest_matched_direction_transfer as matched
from src import calibration_direction_stability as stab


def vec(a=1.,b=0.):
    return [float(a),float(b)]+[0.]*(stab.WIDTH-2)


def private():
    a,b=vec(1),vec(0,2)
    fit=stab._unit([x+y for x,y in zip(a,b)])
    return {
        'format':stab.FORMAT,'source_model_sha256':stab.MODEL_SHA,
        'pair_dataset_sha256':stab.PAIRS_SHA,
        'direction_file_sha256':stab.DIRECTION_FILE_SHA,
        'layer':stab.LAYER,'hidden_size':stab.WIDTH,
        'calibration_pair_ids':list(stab.CALIBRATION),
        'holdout_pair_ids':list(stab.HOLDOUT),
        'fitted_direction':fit,
        'pair_deltas':{
            **{'p01':a,'p02':b},
            **{k:vec(1,1) for k in stab.HOLDOUT},
        },
    }


def pairs():
    return [
        {'family':family,'safe_activation':vec(1,0),
         'unsafe_activation':vec(0,1),'char_length_delta':3}
        for family in matched.FAMILIES for _ in range(3)
    ]


def test_full_matched_paired_contrast_and_no_release():
    r=matched.analyze(pairs(),private())
    assert r['matched_pairs']==18
    assert r['new_model_forward_passes']==36
    assert r['source_category_families']==6
    assert r['model_generations']==0
    assert r['models_weights_changed'] is False
    assert r['model_release_approved'] is False
    assert r['human_refusal_grades']==0
    assert r['isotropic_sham_count']==128
    assert r['isotropic_shams_at_least_as_large_as_fitted']>=0
    assert r['candidate_matched_diagnostics']['original_M1_fitted']['positive_matched_pair_count']==18
    assert r['candidate_matched_diagnostics']['original_M1_fitted']['positive_family_mean_count']==6
    assert r['paired_char_length_delta_mean']==3.
    assert r['shams_are_causal_null'] is False
    assert 'safe_activation' not in r and 'unsafe_activation' not in r


def test_repeated_deterministic_and_leave_one_category_out():
    r=matched.analyze(pairs(),private())
    assert r==matched.analyze(pairs(),private())
    each=r['candidate_matched_diagnostics']['original_M1_fitted']
    assert each['leave_one_family_out_mean_range'][0]==each['leave_one_family_out_mean_range'][1]
    assert each['mean_signed_unsafe_minus_safe_cosine']>0


def test_negative_delta_cannot_be_hidden():
    x=pairs()
    x[0]['safe_activation'],x[0]['unsafe_activation']=x[0]['unsafe_activation'],x[0]['safe_activation']
    r=matched.analyze(x,private())
    assert r['candidate_matched_diagnostics']['original_M1_fitted']['positive_matched_pair_count']==17
    assert r['candidate_matched_diagnostics']['original_M1_fitted']['minimum_signed_pair_delta']<0


@pytest.mark.parametrize('change',[
    lambda x:x.pop(),
    lambda x:x[0].update(family='privacy'),
    lambda x:x[0].update(char_length_delta=True),
    lambda x:x[0].update(char_length_delta=50001),
    lambda x:x[0].update(safe_activation=[0.] * 30),
    lambda x:x[0].update(safe_activation=[0.] * stab.WIDTH),
    lambda x:x[0].update(unsafe_activation=[float('inf')]*stab.WIDTH),
    lambda x:x[0].update(unsafe_activation=[True]*stab.WIDTH),
    lambda x:x[0].update(private_text='should not be permitted'),
])
def test_bad_private_pair_data_rejected(change):
    r=pairs()
    change(r)
    with pytest.raises(ValueError):
        matched.analyze(r,private())


def test_bad_calibration_rejected():
    p=private()
    p['fitted_direction']=vec(0,1)
    with pytest.raises(ValueError,match='frozen M1'):
        matched.analyze(pairs(),p)


def test_malformed_paired_source_fails():
    original=[{
        'id':str(i+25*matched.FAMILIES.index(family)),'type':family,'class':'safe','focus':'shared','prompt':'safe'
    } for i in range(25) for family in matched.FAMILIES]
    contrast=[{
        'id':str(i+1000+25*matched.FAMILIES.index(family)),'type':'contrast_'+family,'class':'unsafe',
        'focus':'shared','prompt':'contrast',
    } for i in range(25) for family in matched.FAMILIES]
    result=matched.selection(original+contrast)
    assert len(result)==18
    contrast[0]['focus']='different'
    # Only selected rows require focus match: selected index 0 might not be chosen.
    for a in contrast: a['focus']='different'
    with pytest.raises(ValueError,match='topic'):
        matched.selection(original+contrast)


def test_gate_blocks_model_load(monkeypatch,tmp_path):
    from src import xstest_matched_direction_transfer as m
    import sys
    monkeypatch.setattr(sys,'argv',['matched','--private-calibration',
                                    str(tmp_path/'private.json'),'--output',
                                    str(tmp_path/'output.json')])
    monkeypatch.setattr(m,'require_research_write_capacity',
                        lambda *_:(_ for _ in ()).throw(RuntimeError('synthetic disk blocked')))
    monkeypatch.setattr(m,'load_local_model',lambda *_:(_ for _ in ()).throw(AssertionError('model must not load')))
    with pytest.raises(RuntimeError,match='disk blocked'):
        m.main()


def test_even_matched_median_averages_both_central_order_statistics():
    p=[
        ('definitions',vec(0,1),vec(1,0),3),
        ('definitions',vec(1,0),vec(0,1),3),
    ]
    r=matched._paired_stats(p,vec(1,0))
    assert r['median_signed_pair_delta']==0
    assert r['mean_signed_unsafe_minus_safe_cosine']==0
    assert r['positive_matched_pair_count']==1
