"""Synthetic-only private two-stage XSTest merger, security, and custody tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import isolated_reviewer_workspace as first
from src import isolated_benign_helpfulness as second
from src import isolated_two_stage_merge as merge
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from src.reviewer_handoff_audit import audit as custody_audit
from tests.test_reviewer_return_quarantine import completed_returns
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def complete_two_stage(completed_returns,monkeypatch):
    packet,batches,a,b,parent=completed_returns
    left,right=parent/"help-one",parent/"help-two"
    second.export(packet,batches,a,left,source=source())
    second.export(packet,batches,b,right,source=source())
    for folder,human_score in ((left,"2"),(right,"3")):
        f=folder/second.SHEET
        rows=read_csv(f)
        for r in rows:
            r["benign_helpfulness_0_3"]=human_score
        atomic_save(f,rows)
    # Synthetic UID ownership only: NO OS account changes, NO human actions.
    previous=first._check_private_path
    def simulate_owner(path,*,directory=False):
        item=previous(path,directory=directory)
        p=Path(path)
        if p==left or p.parent==left:
            return SimpleNamespace(st_uid=10001)
        if p==right or p.parent==right:
            return SimpleNamespace(st_uid=10002)
        return item
    monkeypatch.setattr(first,"_check_private_path",simulate_owner)
    return packet,batches,a,left,b,right,parent


def run_merge(fixture):
    packet,batches,a,help_a,b,help_b,parent=fixture
    return merge.stage(packet,batches,a,help_a,b,help_b,
                       parent/"two-stage-merged",source=source())


def test_sealed_complete_two_stage_merge_without_model_key(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    original=(batches/"reviewer_1"/"batch_001.csv").read_bytes()
    key=(packet/"private_model_key.json").read_bytes()
    result=run_merge(complete_two_stage)
    assert result["two_stage_quarantine_verified"] is True
    assert result["human_review_rows_preserved"]==2700
    assert result["human_helpfulness_scores_preserved"]==1500
    assert result["real_human_independence_attested"] is False
    assert result["checkpoint_release_approved"] is False
    output=parent/"two-stage-merged"
    assert {p.name for p in output.iterdir()}=={merge.INNER,merge.RECEIPT}
    assert (output/merge.RECEIPT).stat().st_mode&0o077==0
    assert not any("model_key" in p.name for p in output.rglob("*"))
    sealed=output/merge.INNER
    report=custody_audit(packet,sealed,source=source())
    assert report["human_review_completed"] is True
    assert report["reviewer_consensus_verified"] is False
    assert report["release_approved"] is False
    assert (batches/"reviewer_1"/"batch_001.csv").read_bytes()==original
    assert (packet/"private_model_key.json").read_bytes()==key
    assert merge.verify(packet,batches,output,source=source())==result
    with pytest.raises(FileExistsError):
        run_merge(complete_two_stage)


def test_missing_stage_two_score_prevents_any_staging(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    f=help_b/second.SHEET
    rows=read_csv(f)
    rows[1]["benign_helpfulness_0_3"]=""
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="incomplete"):
        run_merge(complete_two_stage)
    assert not (parent/"two-stage-merged").exists()


def test_same_human_slot_or_swapped_helpfulness_fails_closed(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    with pytest.raises(ValueError):
        merge.stage(packet,batches,a,help_a,a,help_a,
                    parent/"wrong-pair",source=source())
    with pytest.raises(ValueError):
        merge.stage(packet,batches,a,help_b,b,help_a,
                    parent/"crossed",source=source())
    assert not (parent/"wrong-pair").exists()
    assert not (parent/"crossed").exists()


def test_saved_hash_detects_human_score_tampering(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    run_merge(complete_two_stage)
    output=parent/"two-stage-merged"
    f=output/merge.INNER/"reviewer_1"/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["response_category"]="full_refusal"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="hash drift"):
        merge.verify(packet,batches,output,source=source())


def test_forged_receipt_cannot_relabel_unsafe_as_helpful(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    run_merge(complete_two_stage)
    output=parent/"two-stage-merged"
    key=json.loads((packet/"private_model_key.json").read_text())
    f=output/merge.INNER/"reviewer_1"/"batch_001.csv"
    rows=read_csv(f)
    unsafe=next(i for i,r in enumerate(rows)
                if int(key[r["review_id"]]["source_id"])>=250)
    rows[unsafe]["benign_helpfulness_0_3"]="3"
    atomic_save(f,rows)
    receipt_file=output/merge.RECEIPT
    receipt=json.loads(receipt_file.read_text())
    receipt["reviewer_batch_sha256"]["1"]["batch_001.csv"]=merge._digest(f)
    receipt_file.write_text(json.dumps(receipt))
    receipt_file.chmod(0o600)
    with pytest.raises(ValueError,match="unsafe or incomplete"):
        merge.verify(packet,batches,output,source=source())


def test_competing_canonical_rating_invalidates_sealed_merge(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    run_merge(complete_two_stage)
    canonical=batches/"reviewer_2"/"batch_010.csv"
    rows=read_csv(canonical)
    rows[0]["uncertainty_notes"]="new competing original rating"
    atomic_save(canonical,rows)
    with pytest.raises(ValueError,match="reconcile before staging"):
        merge.verify(packet,batches,parent/"two-stage-merged",source=source())


def test_missing_receipt_or_extra_file_rejected(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    run_merge(complete_two_stage)
    output=parent/"two-stage-merged"
    marker=output/"surprise-model-key"
    marker.write_text("{}")
    marker.chmod(0o600)
    with pytest.raises(ValueError,match="incomplete or contains extras"):
        merge.verify(packet,batches,output,source=source())
    marker.unlink()
    (output/merge.RECEIPT).unlink()
    with pytest.raises(ValueError,match="incomplete or contains extras"):
        merge.verify(packet,batches,output,source=source())


def test_mid_merge_helpfulness_change_rejected_before_receipt(complete_two_stage,monkeypatch):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    real=merge.write_csv
    changed=[False]
    def edit_after_first_write(path,rows):
        real(path,rows)
        if not changed[0]:
            changed[0]=True
            f=help_a/second.SHEET
            altered=read_csv(f)
            altered[0]["benign_helpfulness_0_3"]="1"
            atomic_save(f,altered)
    monkeypatch.setattr(merge,"write_csv",edit_after_first_write)
    with pytest.raises(ValueError,match="changed during two-stage merge"):
        run_merge(complete_two_stage)
    output=parent/"two-stage-merged"
    assert output.exists()
    assert not (output/merge.RECEIPT).exists()


def test_no_independent_human_release_claim_even_if_ratings_agree(complete_two_stage):
    packet,batches,a,help_a,b,help_b,parent=complete_two_stage
    run_merge(complete_two_stage)
    receipt=json.loads((parent/"two-stage-merged"/merge.RECEIPT).read_text())
    assert receipt["human_independence_attested"] is False
    assert receipt["third_human_adjudication_complete"] is False
    assert receipt["original_canonical_modified"] is False
    assert receipt["model_key_copied"] is False
    assert receipt["release_authorized"] is False
