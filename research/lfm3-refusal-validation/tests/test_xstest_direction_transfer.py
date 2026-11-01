"""Synthetic-only tests for independent XSTest activation transfer controls."""
import copy
import math
import pytest

from src import xstest_direction_transfer as transfer
from src import calibration_direction_stability as stable


def vec(x,y=0.):
    return [float(x),float(y)] + [0.]*(stable.WIDTH-2)


def private_source():
    a,b=vec(1),vec(0,2)
    d=stable._unit([x+y for x,y in zip(a,b)])
    return {
        'format':stable.FORMAT,
        'source_model_sha256':stable.MODEL_SHA,
        'pair_dataset_sha256':stable.PAIRS_SHA,
        'direction_file_sha256':stable.DIRECTION_FILE_SHA,
        'layer':stable.LAYER,'hidden_size':stable.WIDTH,
        'calibration_pair_ids':list(stable.CALIBRATION),
        'holdout_pair_ids':list(stable.HOLDOUT),
        'fitted_direction':d,
        'pair_deltas':{
            **{'p01':a,'p02':b},
            **{key:vec(1,1) for key in stable.HOLDOUT},
        },
    }


def toy():
    return [
        {'class':'safe','type':'safe_1','activation':vec(1,0)},
        {'class':'safe','type':'safe_1','activation':vec(1,0)},
        {'class':'unsafe','type':'unsafe_1','activation':vec(0,1)},
        {'class':'unsafe','type':'unsafe_1','activation':vec(0,1)},
    ]


def test_nonbehavioral_controls_are_deterministic_and_aggregate_only():
    rows=toy()
    result=transfer.aggregate(rows,private_source(),full=False)
    assert result==transfer.aggregate(rows,private_source(),full=False)
    assert result['prompt_count']==4
    assert result['isotropic_sham_count']==128
    assert result['safe_prompts']==2 and result['unsafe_prompts']==2
    assert result['model_generations']==0
    assert result['model_weights_changed'] is False
    assert result['human_refusal_grades']==0
    assert result['model_release_approved'] is False
    assert len(result['candidate_class_cosine'])==4
    assert 0<=result['isotropic_shams_with_abs_gap_at_least_fitted']<=128
    assert 'activation' not in result
    assert 'safe_1' not in str(result)


def test_calibration_equal_weight_differs_from_original_on_synthetic_source():
    result=transfer.aggregate(toy(),private_source(),full=False)
    weighted=result['candidate_class_cosine']['original_M1_fitted']
    balanced=result['candidate_class_cosine']['equal_unit_weight']
    assert balanced['unsafe_minus_safe_cosine'] != weighted['unsafe_minus_safe_cosine']
    assert abs(weighted['unsafe_minus_safe_cosine']) <= 2.


def test_full_study_rejects_short_toy_data():
    with pytest.raises(ValueError,match='stratified'):
        transfer.aggregate(toy(),private_source())


def test_full_study_accepts_exact_category_and_class_counts():
    rows=[]
    for cls,n in [('safe',10),('unsafe',8)]:
        for i in range(n):
            for _ in range(3):
                rows.append({'class':cls,'type':f'{cls}_category_{i}',
                             'activation':vec(1 if cls=='safe' else 0,
                                              0 if cls=='safe' else 1)})
    result=transfer.aggregate(rows,private_source())
    assert result['prompt_count']==54
    assert result['category_count']==18
    assert result['safe_prompts']==30 and result['unsafe_prompts']==24


@pytest.mark.parametrize('change',[
    lambda x:x[0].update(activation=[0.] * 7),
    lambda x:x[0].update(activation=['bad'] * stable.WIDTH),
    lambda x:x[0].update(activation=[float('nan')] * stable.WIDTH),
    lambda x:x[0].update(activation=[0.] * stable.WIDTH),
    lambda x:x[0].update(activation=[True] * stable.WIDTH),
    lambda x:x[0].update(type=''),
    lambda x:x[0].update(**{'class':'unknown'}),
    lambda x:(x.pop(1),x.pop(0)),
])
def test_bad_activation_class_or_source_rejected(change):
    rows=toy()
    change(rows)
    with pytest.raises(ValueError):
        transfer.aggregate(rows,private_source(),full=False)


def test_tampered_calibrator_rejected():
    a=private_source()
    a['source_direction_file_sha256']='0'*64
    with pytest.raises(ValueError):
        transfer.aggregate(toy(),a,full=False)


def test_capacity_preflight_blocks_model_loading(monkeypatch,tmp_path):
    from src import xstest_direction_transfer as mod
    import sys
    monkeypatch.setattr(sys,'argv',['xstest_direction_transfer',
        '--private-calibration',str(tmp_path/'private.json'),
        '--output',str(tmp_path/'output.json')])
    def block(*_):
        raise RuntimeError('synthetic root block')
    monkeypatch.setattr(mod,'require_research_write_capacity',block)
    monkeypatch.setattr(mod,'load_local_model',lambda *x: (_ for _ in ()).throw(
        AssertionError('cannot load model when root is blocked')))
    with pytest.raises(RuntimeError,match='root block'):
        mod.main()
    assert not (tmp_path/'output.json').exists()
