"""Synthetic-only fail-closed quarantine tests; never real human grading."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import isolated_reviewer_workspace as isolated
from src import reviewer_return_quarantine as quarantine
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv,split_packet
from tests.test_reviewer_batches import create_packet
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def completed_returns(tmp_path,monkeypatch):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    canonical=tmp_path/"canonical"
    split_packet(packet,canonical)
    parent=tmp_path/"private_staging"
    parent.mkdir(mode=0o700)
    first,second=parent/"first",parent/"second"
    isolated.export(packet,canonical,1,first,source=source())
    isolated.export(packet,canonical,2,second,source=source())
    for path,username in ((first,"synthetic-person-one"),(second,"synthetic-person-two")):
        for i in range(1,16):
            f=path/f"batch_{i:03d}.csv"
            records=read_csv(f)
            for row in records:
                row.update(reviewer_id=username,
                           response_category="full_compliance",
                           actionable_harmful_content="no")
            atomic_save(f,records)
    # Only synthetic tests simulate distinct Unix ownership.
    orig=isolated._check_private_path
    def fake(path,*,directory=False):
        st=orig(path,directory=directory)
        p=Path(path)
        if p==first or p.parent==first:
            return SimpleNamespace(st_uid=10001)
        if p==second or p.parent==second:
            return SimpleNamespace(st_uid=10002)
        return st
    monkeypatch.setattr(isolated,"_check_private_path",fake)
    return packet,canonical,first,second,parent


def stage_synthetic(inputs):
    packet,canonical,first,second,parent=inputs
    return quarantine.stage(packet,canonical,first,second,
                            parent/"returned_quarantine",source=source())


def test_two_returns_quarantined_without_changing_originals(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    frozen_source=(packet/"blinded_reviewer_1.csv").read_bytes()
    frozen_key=(packet/"private_model_key.json").read_bytes()
    frozen_batch=(canonical/"reviewer_1"/"batch_001.csv").read_bytes()
    result=stage_synthetic(completed_returns)
    assert result["stage_one_quarantine_sealed"] is True
    assert result["blinded_human_rating_rows_preserved"]==2700
    assert result["reviewer_batch_count"]==30
    assert result["original_source_context_verified"] is True
    assert result["model_key_included"] is False
    assert result["actual_human_independence_attested"] is False
    assert result["stage_two_helpfulness_completed"] is False
    assert result["canonical_review_packets_modified"] is False
    assert result["release_authorized"] is False
    out=parent/"returned_quarantine"
    assert {p.name for p in out.iterdir()}=={
        quarantine.BATCHDIR,quarantine.RECEIPT}
    assert not any("model_key" in p.name for p in out.rglob("*"))
    assert (out/quarantine.RECEIPT).stat().st_mode&0o077==0
    assert (packet/"blinded_reviewer_1.csv").read_bytes()==frozen_source
    assert (packet/"private_model_key.json").read_bytes()==frozen_key
    assert (canonical/"reviewer_1"/"batch_001.csv").read_bytes()==frozen_batch
    assert quarantine.verify_quarantine(packet,canonical,out,source=source())==result
    with pytest.raises(FileExistsError):
        stage_synthetic(completed_returns)


def test_same_owner_return_cannot_be_staged_even_with_completed_ratings(tmp_path):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    batches=tmp_path/"batches"
    split_packet(packet,batches)
    private=tmp_path/"private"
    private.mkdir(mode=0o700)
    first,second=private/"first",private/"second"
    isolated.export(packet,batches,1,first,source=source())
    isolated.export(packet,batches,2,second,source=source())
    with pytest.raises(PermissionError,match="distinct non-root"):
        quarantine.stage(packet,batches,first,second,private/"out",source=source())
    assert not (private/"out").exists()


def test_quarantine_detects_modified_and_extra_review_data(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    out=parent/"returned_quarantine"
    f=out/"batches"/"reviewer_1"/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["response_category"]="full_refusal"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="evidence modified"):
        quarantine.verify_quarantine(packet,canonical,out,source=source())
    with pytest.raises(FileExistsError):
        stage_synthetic(completed_returns)


def test_quarantine_rejects_missing_receipt_and_unexpected_files(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    out=parent/"returned_quarantine"
    receipt=out/quarantine.RECEIPT
    receipt.unlink()
    with pytest.raises(ValueError,match="incomplete/extra"):
        quarantine.verify_quarantine(packet,canonical,out,source=source())
    with receipt.open("x",encoding="utf-8") as f:
        f.write("{}")
    receipt.chmod(0o600)
    with pytest.raises(ValueError,match="invalid quarantine"):
        quarantine.verify_quarantine(packet,canonical,out,source=source())


def test_no_disclosure_in_quarantine_receipt(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    out=parent/"returned_quarantine"
    receipt=(out/quarantine.RECEIPT).read_text()
    data=json.loads(receipt)
    assert "synthetic-person" not in receipt
    assert "model_response" not in receipt and "prompt" not in receipt
    assert "source_id" not in receipt and '"arm"' not in receipt
    assert data["independent_real_humans_attested"] is False
    assert data["release_authorized"] is False
    assert len(data["reviewer_batch_sha256"]["1"])==15
    assert len(data["reviewer_batch_sha256"]["2"])==15


def test_quarantine_rejects_output_inside_input_folder(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    with pytest.raises(ValueError,match="cannot be created inside"):
        quarantine.stage(packet,canonical,first,second,
                         first/"bad-output",source=source())
    assert not (first/"bad-output").exists()


def test_import_is_not_adjudicated_and_safe_helpfulness_not_faked(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    out=parent/"returned_quarantine"
    from src.reviewer_handoff_audit import audit
    result=audit(packet,out/"batches",source=source())
    assert result["human_review_completed"] is False
    assert result["reviewer_consensus_verified"] is False
    assert result["release_approved"] is False
    # All 750 synthetic benign cases per reviewer require phase-two grading.
    assert result["reviewer_1"]["ungraded"]==750
    assert result["reviewer_2"]["ungraded"]==750


def test_resealed_receipt_cannot_turn_missing_rating_into_valid_evidence(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    root=parent/"returned_quarantine"
    f=root/"batches"/"reviewer_2"/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["response_category"]=""
    atomic_save(f,rows)
    receipt_file=root/quarantine.RECEIPT
    receipt=json.loads(receipt_file.read_text())
    receipt["reviewer_batch_sha256"]["2"]["batch_001.csv"]=quarantine._sha(f)
    receipt_file.write_text(json.dumps(receipt))
    receipt_file.chmod(0o600)
    with pytest.raises(ValueError,match="incomplete or invalid quarantined"):
        quarantine.verify_quarantine(packet,canonical,root,source=source())


def test_failed_second_reviewer_copy_never_creates_sealed_receipt(
        completed_returns,monkeypatch):
    packet,canonical,first,second,parent=completed_returns
    original=quarantine._copy_and_verify
    attempts=iter((1,2))
    def interrupted(*args,**kwargs):
        if next(attempts)==2:
            raise OSError("synthetic disk interruption")
        return original(*args,**kwargs)
    monkeypatch.setattr(quarantine,"_copy_and_verify",interrupted)
    with pytest.raises(OSError,match="interruption"):
        stage_synthetic(completed_returns)
    output=parent/"returned_quarantine"
    assert output.is_dir()
    assert not (output/quarantine.RECEIPT).exists()
    with pytest.raises(ValueError,match="incomplete/extra"):
        quarantine.verify_quarantine(packet,canonical,output,source=source())
    assert (packet/"private_model_key.json").exists()


def test_quarantine_detects_changed_canonical_source_after_seal(completed_returns):
    packet,canonical,first,second,parent=completed_returns
    stage_synthetic(completed_returns)
    origin=canonical/"reviewer_1"/"batch_001.csv"
    rows=read_csv(origin)
    rows[0]["model_response"]="modified after quarantine"
    atomic_save(origin,rows)
    with pytest.raises(ValueError):
        quarantine.verify_quarantine(packet,canonical,parent/"returned_quarantine",source=source())
