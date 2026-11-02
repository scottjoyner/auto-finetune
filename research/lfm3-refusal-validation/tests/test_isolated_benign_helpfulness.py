"""Synthetic-only key-free second-stage reviewer eligibility and custody tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import isolated_reviewer_workspace as first
from src import isolated_benign_helpfulness as second
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from tests.test_reviewer_return_quarantine import completed_returns
from tests.test_reviewer_handoff_audit import source


def create_stage_two(completed_returns):
    packet,batches,one,two,parent=completed_returns
    target=parent/"benign-stage-two"
    result=second.export(packet,batches,one,target,source=source())
    return packet,batches,one,two,parent,target,result


def test_not_available_until_full_phase_one(completed_returns):
    packet,batches,one,two,parent=completed_returns
    rows=read_csv(one/"batch_004.csv")
    rows[5]["actionable_harmful_content"]=""
    atomic_save(one/"batch_004.csv",rows)
    with pytest.raises(ValueError,match="incomplete"):
        second.export(packet,batches,one,parent/"benign",source=source())
    assert not (parent/"benign").exists()


def test_only_750_safe_responses_and_no_arm_key_exported(completed_returns):
    packet,batches,one,two,parent,target,result=create_stage_two(completed_returns)
    assert result["eligible_benign_model_responses"]==750
    assert result["first_stage_human_ratings_verified"]==1350
    assert result["actual_human_helpfulness_scores_entered"]==0
    assert result["model_identity_unblinded"] is False
    assert {p.name for p in target.iterdir()}=={second.MANIFEST,second.SHEET}
    m=json.loads((target/second.MANIFEST).read_text())
    assert m["model_key_copied"] is False
    assert m["reviewer_slot"]==1
    assert m["human_identity_attested"] is False
    assert "arm" not in m and "source_id" not in m
    ids={r["review_id"] for r in read_csv(target/second.SHEET)}
    key=json.loads((packet/"private_model_key.json").read_text())
    assert len(ids)==750
    assert all(int(key[r]["source_id"])<250 for r in ids)
    assert len({key[r]["arm"] for r in ids})==3
    assert all(r["benign_helpfulness_0_3"]=="" for r in read_csv(target/second.SHEET))
    assert second.inspect(target,one)["separate_reviewer_unix_owner"] is False
    assert second.inspect(target,one)["human_helpfulness_remaining"]==750
    assert (target/second.SHEET).stat().st_mode&0o077==0
    with pytest.raises(FileExistsError):
        second.export(packet,batches,one,target,source=source())


def test_same_exporter_cannot_perform_helpfulness_review(completed_returns,monkeypatch):
    packet,batches,one,two,parent,target,_=create_stage_two(completed_returns)
    monkeypatch.setattr(second.os,"isatty",lambda _:True)
    with pytest.raises(PermissionError,match="original non-root reviewer"):
        second.review(target,one,"synthetic-person-one")


def test_changed_phase_one_human_response_denies_phase_two(completed_returns):
    packet,batches,one,two,parent,target,_=create_stage_two(completed_returns)
    path=one/"batch_007.csv"
    rows=read_csv(path)
    rows[3]["uncertainty_notes"]="late edit"
    atomic_save(path,rows)
    with pytest.raises(ValueError,match="changed since benign export"):
        second.inspect(target,one)


def test_stage_two_extra_files_and_changed_responses_refused(completed_returns):
    packet,batches,one,two,parent,target,_=create_stage_two(completed_returns)
    extra=target/"private_model_key.json"
    extra.write_text("{}")
    extra.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected"):
        second.inspect(target,one)
    extra.unlink()
    f=target/second.SHEET
    rows=read_csv(f)
    rows[0]["model_response"]="tampered synthetic model response"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="frozen ratings changed"):
        second.inspect(target,one)


def test_wrong_reviewer_work_and_unsafe_grades_not_accepted(completed_returns):
    packet,batches,one,two,parent,target,_=create_stage_two(completed_returns)
    with pytest.raises(ValueError,match="wrong original reviewer|changed since benign export"):
        second.inspect(target,two)
    f=target/second.SHEET
    rows=read_csv(f)
    key=json.loads((packet/"private_model_key.json").read_text())
    unsafe=next(k for k,v in key.items() if int(v["source_id"])>=250)
    rows[0]["review_id"]=unsafe
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="changed"):
        second.inspect(target,one)


def test_human_score_append_only_and_750_return_custody(completed_returns,monkeypatch):
    packet,batches,one,two,parent,target,_=create_stage_two(completed_returns)
    original=first._check_private_path
    def synthetic_handoff(path,*,directory=False):
        stat_result=original(path,directory=directory)
        p=Path(path)
        if p==target or p.parent==target:
            return SimpleNamespace(st_uid=10001)
        return stat_result
    monkeypatch.setattr(first,"_check_private_path",synthetic_handoff)
    # Human responses are SYNTHETIC; no real rater exists in this test.
    f=target/second.SHEET
    rows=read_csv(f)
    from src.reviewer_helpfulness_pass import review_one_helpfulness
    answers=iter(("2",))
    assert review_one_helpfulness(f,rows,0,"synthetic-person-one",set(r["review_id"] for r in rows),
                                 input_fn=lambda _:next(answers),output_fn=lambda _:None)
    with pytest.raises(ValueError,match="cannot be replaced"):
        review_one_helpfulness(f,read_csv(f),0,"synthetic-person-one",
                              set(r["review_id"] for r in rows),
                              input_fn=lambda _: "3",output_fn=lambda _:None)
    state=second.inspect(target,one)
    assert state["human_helpfulness_ratings_entered"]==1
    assert state["human_helpfulness_remaining"]==749
    with pytest.raises(ValueError,match="incomplete"):
        second.verify_return(packet,batches,one,target,source=source())
    allrows=read_csv(f)
    for r in allrows:
        if not r["benign_helpfulness_0_3"]:
            r["benign_helpfulness_0_3"]="2"
    atomic_save(f,allrows)
    result=second.verify_return(packet,batches,one,target,source=source())
    assert result["human_benign_helpfulness_ratings_verified"]==750
    assert result["source_benchmark_safe_eligibility_verified"] is True
    assert result["frozen_first_stage_human_labels_unchanged"] is True
    assert result["independent_person_identity_verified"] is False
    assert result["canonical_files_modified"] is False
    assert result["release_authorized"] is False


def test_defers_stage_two_if_real_reviewers_not_separated(tmp_path):
    from src.reviewer_batches import split_packet
    from tests.test_reviewer_batches import create_packet
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    batches=tmp_path/"batches"
    split_packet(packet,batches)
    owner=tmp_path/"stage-one"
    owner.mkdir(mode=0o700)
    with pytest.raises(Exception):
        second.export(packet,batches,owner,tmp_path/"benign",source=source())
    assert not (tmp_path/"benign").exists()
