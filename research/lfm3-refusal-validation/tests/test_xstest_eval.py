from collections import Counter
from types import SimpleNamespace
import pytest

from src.xstest_450_eval import (
    manifest_for, stratified_selection, verify_resume, record_summary
)


def fake_rows():
    return [dict(id=f'{c}-{i}', type=c, prompt='harmless placeholder',
                 **{'class': 'unsafe' if c.startswith('contrast_') else 'safe'})
            for c in ('safe_contexts', 'contrast_safe_contexts', 'definitions',
                      'contrast_definitions') for i in range(25)]


def test_selection_is_balanced_and_reproducible():
    dataset = fake_rows()
    a = stratified_selection(dataset, 3, 19)
    b = stratified_selection(dataset, 3, 19)
    assert len(a) == 12
    assert [r['id'] for r in a] == [r['id'] for r in b]
    assert Counter(x['type'] for x in a) == {x: 3 for x in ('safe_contexts', 'contrast_safe_contexts', 'definitions', 'contrast_definitions')}


def test_full_selection_and_bad_limit():
    assert len(stratified_selection(fake_rows(), 0, 17)) == 100
    with pytest.raises(ValueError):
        stratified_selection(fake_rows(), 26, 17)


def test_duplicate_result_rejected():
    rows = fake_rows()[:1]
    completed = [dict(arm='A_original_posttrained', id=rows[0]['id'])]
    with pytest.raises(ValueError, match='duplicate'):
        verify_resume(rows, completed * 2)


def test_unknown_arm_rejected():
    rows = fake_rows()[:1]
    with pytest.raises(ValueError, match='unknown'):
        verify_resume(rows, [dict(arm='other', id=rows[0]['id'])])


def test_proxy_summary_not_safety_score():
    sel=[dict(id='a', type='safe_contexts', **{'class':'safe'}),
         dict(id='b', type='contrast_safe_contexts', **{'class':'unsafe'})]
    rows = [dict(arm='A_original_posttrained',id='a',refusal_proxy=False),
            dict(arm='A_original_posttrained',id='b',refusal_proxy=True)]
    got = record_summary(rows,sel)
    assert got['A_original_posttrained']['safe']['complete_generations'] == 1
    assert got['A_original_posttrained']['unsafe']['lexical_refusal_count'] == 1
    assert got['B_M1_abliterated_smoke']['unsafe']['lexical_refusal_rate'] is None
