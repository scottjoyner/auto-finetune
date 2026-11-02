"""Synthetic only: provenance-bound, key-free third human adjudication handoff."""
import json
from pathlib import Path

import pytest
from src import verified_adjudication_handoff as handoff
from src import isolated_two_stage_merge as merged
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from tests.test_reviewer_return_quarantine import completed_returns
from tests.test_isolated_two_stage_merge import complete_two_stage,run_merge
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def prepared(complete_two_stage):
    packet,canonical,first,help_first,second,help_second,parent=complete_two_stage
    run_merge(complete_two_stage)
    two_stage=parent/"two-stage-merged"
    escrow=parent/"third-human-escrow"
    result=handoff.prepare(packet,canonical,two_stage,escrow,first,second,source=source())
    return packet,canonical,two_stage,escrow,result,first,second


def test_escrow_prepares_blinded_disputes_and_never_claims_personhood(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    assert result["verified_source_selection"] is True
    assert result["source_reviewed_model_prompt_pairs"]==1350
    assert result["disputed_or_unresolved_human_items"]==750
    assert result["human_first_stage_adjudications_entered"]==0
    assert result["third_human_workspace_distinct_unix_owner"] is False
    assert result["original_two_humans_independently_attested"] is False
    assert result["human_consensus_verified"] is False
    assert result["release_authorized"] is False
    assert {p.name for p in escrow.iterdir()}=={
        handoff.RECEIPT,handoff.STAGING}
    ws=escrow/handoff.STAGING
    assert {p.name for p in ws.iterdir()}=={handoff.MANIFEST,handoff.SHEET}
    assert all(not any(k in p.name for k in ("model_key","reviewer_1","reviewer_2"))
               for p in ws.iterdir())
    assert (ws/handoff.SHEET).stat().st_mode&0o077==0
    assert (escrow/handoff.RECEIPT).stat().st_mode&0o077==0
    labels=read_csv(ws/handoff.SHEET)
    assert len(labels)==750
    assert all(r["reviewer_id"]=="" and r["response_category"]==""
               and r["actionable_harmful_content"]==""
               and r["benign_helpfulness_0_3"]==""
               for r in labels)
    text=(ws/handoff.MANIFEST).read_text()
    assert "source_id" not in text and '"arm"' not in text
    assert "private_model_key" not in text and "synthetic-person" not in text
    assert handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())==result


def test_preparing_twice_refuses_overwrite(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    with pytest.raises(FileExistsError):
        handoff.prepare(packet,canonical,two_stage,escrow,first,second,source=source())


def test_edited_dispute_prompt_rejected(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    f=escrow/handoff.STAGING/handoff.SHEET
    rows=read_csv(f)
    rows[0]["model_response"]="poisoned synthetic answer"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="source items"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_adjudicator_first_stage_rejects_helpfulness_without_safe_unlock(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    f=escrow/handoff.STAGING/handoff.SHEET
    rows=read_csv(f)
    rows[0].update(reviewer_id="synthetic-third",response_category="full_refusal",
                   actionable_harmful_content="no",benign_helpfulness_0_3="2")
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="cannot disclose benign"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_untrusted_or_missing_escrow_receipt_refused(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    receipt=escrow/handoff.RECEIPT
    r=json.loads(receipt.read_text())
    r["release_authorized"]=True
    receipt.write_text(json.dumps(r))
    receipt.chmod(0o600)
    with pytest.raises(ValueError,match="invalid third-human escrow"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())
    receipt.unlink()
    with pytest.raises(ValueError,match="incomplete/extra"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_merged_source_change_invalidates_third_human_handoff(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    f=two_stage/merged.INNER/"reviewer_2"/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["response_category"]="partial_refusal"
    atomic_save(f,rows)
    with pytest.raises(ValueError):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_no_adjudicator_grades_synthetically_filled_by_prepare(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    assert result["human_first_stage_adjudications_entered"]==0
    assert all(not r["reviewer_id"].strip() for r in
               read_csv(escrow/handoff.STAGING/handoff.SHEET))


def test_reviewer_grades_allowed_only_as_initial_without_helpfulness(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    f=escrow/handoff.STAGING/handoff.SHEET
    rows=read_csv(f)
    rows[0].update(reviewer_id="synthetic-third",response_category="full_refusal",
                   actionable_harmful_content="no")
    atomic_save(f,rows)
    status=handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())
    assert status["human_first_stage_adjudications_entered"]==1
    assert status["third_human_identity_independently_attested"] is False
    assert status["human_adjudication_fully_complete"] is False
    assert status["release_authorized"] is False
    rows[1]["reviewer_id"]="different-third"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="changed rater identity"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_same_owner_third_human_interactive_review_denied(prepared,monkeypatch):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    monkeypatch.setattr(handoff.os,"isatty",lambda _:True)
    with pytest.raises(PermissionError,match="separately owned"):
        handoff.review(escrow,escrow/handoff.STAGING,"synthetic-third")


def test_original_reviewer_cannot_masquerade_as_third_human(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    f=escrow/handoff.STAGING/handoff.SHEET
    rows=read_csv(f)
    rows[0].update(
        reviewer_id="synthetic-person-one",
        response_category="full_refusal",
        actionable_harmful_content="no")
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="duplicates an original reviewer"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_escrow_rejects_undeclared_files(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    unexpected=escrow/handoff.STAGING/"private_model_key.json"
    unexpected.write_text("{}")
    unexpected.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected files"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())
    unexpected.unlink()
    unexpected=escrow/"unexpected.json"
    unexpected.write_text("{}")
    unexpected.chmod(0o600)
    with pytest.raises(ValueError,match="incomplete/extra"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())


def test_third_human_cannot_run_under_either_original_reviewer_uid(prepared,monkeypatch):
    from src import isolated_reviewer_workspace as first_module
    from types import SimpleNamespace
    packet,canonical,two_stage,escrow,result,first,second=prepared
    staging=escrow/handoff.STAGING
    original=first_module._check_private_path
    def fake_third_owner(path,*,directory=False):
        st=original(path,directory=directory)
        p=Path(path)
        if p==staging or p.parent==staging:
            return SimpleNamespace(st_uid=10001)
        return st
    monkeypatch.setattr(first_module,"_check_private_path",fake_third_owner)
    monkeypatch.setattr(handoff.os,"isatty",lambda _:True)
    monkeypatch.setattr(handoff.os,"geteuid",lambda:10001)
    with pytest.raises(PermissionError,match="separately owned"):
        handoff.review(escrow,staging,"third-person-new-id")


def test_third_human_escrow_receipt_cannot_rewrite_original_owner_ledger(prepared):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    receipt_path=escrow/handoff.RECEIPT
    receipt=json.loads(receipt_path.read_text())
    assert len(receipt["original_reviewer_uids"])==2
    receipt["original_reviewer_uids"]=[10003,10004]
    receipt_path.write_text(json.dumps(receipt))
    receipt_path.chmod(0o600)
    with pytest.raises(ValueError,match="account custody changed"):
        handoff.verify(packet,canonical,two_stage,escrow,first,second,source=source())
