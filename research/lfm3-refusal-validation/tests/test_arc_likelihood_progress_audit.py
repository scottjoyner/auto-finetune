import pytest
from src.arc_likelihood_progress_audit import validate_journal, validate_manifest, progress_summary
from src.arc_full_likelihood_eval import manifest_for, ORDER


def sample():
    cases=[{"id":"p1","correct_index":0,"options":["X","Y"]},
           {"id":"p2","correct_index":1,"options":["X","Y"]}]
    manifest=manifest_for(cases,{a:str(i) for i,a in enumerate(ORDER)},2)
    return cases, manifest


def row(arm="A_original_posttrained",case="p1",correct=0):
    return {
        "id":case,"arm":arm,"correct_index":correct,
        "predicted_raw_index":0,"predicted_length_normalized_index":0,
        "raw_correct":correct==0,"length_normalized_correct":correct==0,
        "raw_logprobs":[-2.0,-3.0],
        "mean_per_token_logprobs":[-1.0,-1.5],
        "candidate_token_lengths":[2,2],"elapsed_seconds":0.1
    }


def test_manifest_frozen_three_arm():
    cases,m=sample()
    assert validate_manifest(m,cases)
    altered=dict(m,revision="changed")
    with pytest.raises(ValueError,match="revision"):
        validate_manifest(altered,cases)
    with pytest.raises(ValueError,match="local checkpoint"):
        validate_manifest(m,cases,{a:"wrong" for a in ORDER})


def test_valid_partial_and_complete_journal():
    cases,m=sample()
    rows=[row(),row(case="p2",correct=1)]
    assert not validate_journal(m,rows,cases)
    rows += [row(arm=ORDER[1]),row(arm=ORDER[1],case="p2",correct=1)]
    rows += [row(arm=ORDER[2]),row(arm=ORDER[2],case="p2",correct=1)]
    assert validate_journal(m,rows,cases)


def test_audit_rejects_skip_in_prefix():
    cases,m=sample()
    with pytest.raises(ValueError,match="contiguous"):
        validate_journal(m,[row(case="p2",correct=1)],cases)


def test_audit_rejects_changed_answer():
    cases,m=sample()
    with pytest.raises(ValueError,match="correct answer"):
        validate_journal(m,[row(correct=1)],cases)


def test_audit_rejects_score_mismatch():
    cases,m=sample()
    x=row()
    x["predicted_raw_index"]=1
    with pytest.raises(ValueError,match="recorded scores"):
        validate_journal(m,[x],cases)


def test_audit_rejects_nan_and_bad_flags():
    cases,m=sample()
    x=row()
    x["raw_logprobs"]=[float("nan"),-3.0]
    with pytest.raises(ValueError,match="finite"):
        validate_journal(m,[x],cases)
    x=row()
    x["length_normalized_correct"]="true"
    with pytest.raises(ValueError,match="correctness"):
        validate_journal(m,[x],cases)


def test_progress_stays_no_safety_claim():
    cases,m=sample()
    x=progress_summary(m,[row()],False,False,"digest")
    assert not x["safety_outcomes_validated"]
    assert not x["official_arc_score"]
    assert x["per_arm"][ORDER[0]]["remaining"]==1
