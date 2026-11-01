"""Synthetic-only tests for the human matched activation-response bridge.

These are fabricated test annotations, never actual XSTest human outcomes.
"""
import json
import pytest

from src.human_matched_activation_bridge import analyze, METRICS
from src.xstest_matched_direction_transfer import FAMILIES, selection
from src.xstest_450_eval import ARM_ORDER

SAFE_NAMES=('homonyms','figurative_language','safe_targets','safe_contexts',
            'definitions','nons_group_real_discr','real_group_nons_discr',
            'historical_events','privacy_public','privacy_fictional')
UNSAFE_NAMES=('contrast_homonyms','contrast_figurative_language','contrast_safe_targets',
              'contrast_safe_contexts','contrast_definitions','contrast_discr',
              'contrast_historical_events','contrast_privacy')


def fake_benchmark():
    """Exact synthetic 450=250 safe + 200 unsafe source-identities."""
    rows={}
    for i in range(450):
        safe=i<250
        name=(SAFE_NAMES[i//25] if safe else UNSAFE_NAMES[(i-250)//25])
        row={'id':str(i),'type':name,'class':'safe' if safe else 'unsafe',
             'focus':'shared', 'prompt':'synthetic only'}
        rows[str(i)]=row
    return rows


def fake_consensus(source):
    """Synthetic complete 450×3 resolved results; no grader identities."""
    result={}
    for case,row in source.items():
        is_safe=row['class']=='safe'
        for arm in ARM_ORDER:
            result[(case,arm)]={
                'over_refusal':0,
                'full_refusal':0,
                'actionable_harmful_content':0,
                'helpfulness_0_3':2 if is_safe else None,
            }
    return result


def test_synthetic_complete_human_triplets_and_paired_directional_effects():
    source=fake_benchmark()
    labels=fake_consensus(source)
    _, safe, unsafe=selection(list(source.values()))[0]
    a,b,c=ARM_ORDER
    labels[(safe['id'],a)]['over_refusal']=1
    labels[(unsafe['id'],a)]['actionable_harmful_content']=1
    labels[(safe['id'],b)]['helpfulness_0_3']=3
    labels[(unsafe['id'],b)]['full_refusal']=1
    report=analyze(labels,source)
    assert report['source_prompt_pairs']==18
    assert report['source_reviewed_model_prompt_pairs']==1350
    assert report['source_family_count']==6
    assert report['by_arm'][a]['benign_not_over_refused_count']==17
    assert report['by_arm'][a]['unsafe_non_actionable_count']==17
    assert report['by_arm'][a]['pairwise_joint_desirable_count']==17
    assert report['by_arm'][b]['pairwise_joint_desirable_count']==18
    assert report['by_arm'][b]['benign_helpfulness_mean_0_3']==round(2+1/18,6)
    assert len(report['matched_cross_arm_differences'])==15
    pair=next(x for x in report['matched_cross_arm_differences']
              if x['left_arm']==a and x['right_arm']==b and
              x['metric']=='paired_benign_acceptable_and_unsafe_not_actionable')
    assert pair['right_greater_count']==1
    assert pair['left_greater_count']==0
    assert pair['mean_right_minus_left']==round(1/18,6)
    assert report['automatic_model_grades']==0
    assert report['checkpoint_release_approved'] is False
    assert report['familywise_confirmatory_inference_performed'] is False
    public=json.dumps(report)
    assert safe['id'] not in [key for key in report]
    assert 'synthetic only' not in public
    assert 'individual_human_judgments' not in report


def test_no_grade_differences_does_not_claim_improvement():
    source=fake_benchmark()
    out=analyze(fake_consensus(source),source)
    assert len(out['by_arm'])==3
    assert all(x['mean_right_minus_left']==0 for x in out['matched_cross_arm_differences'])
    assert all(x['equal_count']==18 for x in out['matched_cross_arm_differences'])
    assert all(x['p_value_reported'] is False for x in out['matched_cross_arm_differences'])


@pytest.mark.parametrize('change',[
    lambda a,s:a.pop((next(iter(s)),ARM_ORDER[0])),
    lambda a,s:a[(next(iter(s)),ARM_ORDER[0])].update(helpfulness_0_3=None),
    lambda a,s:a[(next(iter(s)),ARM_ORDER[0])].update(over_refusal=True),
    lambda a,s:a[(next(iter(s)),ARM_ORDER[0])].update(full_refusal=1),
    lambda a,s:a.update({('forged-case',ARM_ORDER[1]):{
        'over_refusal':0,'full_refusal':0,
        'actionable_harmful_content':0,'helpfulness_0_3':2}}),
])
def test_incomplete_forged_or_bad_graded_human_inputs_rejected(change):
    benchmark=fake_benchmark()
    labels=fake_consensus(benchmark)
    change(labels,benchmark)
    with pytest.raises(ValueError):
        analyze(labels,benchmark)


def test_source_mapping_change_or_bad_category_is_rejected():
    benchmark=fake_benchmark()
    labels=fake_consensus(benchmark)
    target=next(x for x in benchmark.values() if x['type']=='definitions')
    target['focus']='spoofed'
    with pytest.raises(ValueError,match='topic'):
        # The first changed row need not be sampled; invalidate all source focus.
        for item in benchmark.values():
            if item['type']=='definitions':
                item['focus']='other'
        analyze(labels,benchmark)
