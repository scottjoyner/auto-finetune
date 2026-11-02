"""Synthetic-only private reviewer receipt tests; never attest real human owners."""
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import isolated_reviewer_workspace as isolated
from src import reviewer_transfer_receipt as transfer_receipt
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from tests.test_isolated_reviewer_workspace import originals
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def stage(originals):
    packet,batches,storage=originals
    one=storage/"source-reviewer-one"
    two=storage/"source-reviewer-two"
    isolated.export(packet,batches,1,one,source=source())
    isolated.export(packet,batches,2,two,source=source())
    return packet,batches,one,two,storage


def snapshot(stage):
    return transfer_receipt.snapshot(*stage[:4],source=source())


def verified(stage,receipt):
    return transfer_receipt.verify_staging(
        receipt,*stage[:4],source=source())


def test_live_format_private_receipt_requires_two_pristine_keyfree_exports(stage):
    receipt=snapshot(stage)
    assert receipt["kind"]==transfer_receipt.KIND
    assert receipt["original_blinded_rows_per_reviewer"]==1350
    assert set(receipt["reviewer_1"]["files"])==set(transfer_receipt.FILES)
    assert set(receipt["reviewer_2"]["files"])==set(transfer_receipt.FILES)
    assert receipt["reviewer_1"]["slot"]==1
    assert receipt["reviewer_2"]["slot"]==2
    assert receipt["real_humans_independently_verified"] is False
    assert receipt["os_cross_access_denials_verified"] is False
    assert receipt["model_release_authorized"] is False
    assert "model_response" not in json.dumps(receipt)
    assert "full_refusal" not in json.dumps(receipt)
    assert verified(stage,receipt)["new_transfer_performed"] is False


def test_one_time_owner_only_receipt_and_no_original_changes(stage):
    receipt=snapshot(stage)
    p=stage[4]/"owner-only-transfer-receipt.json"
    result=transfer_receipt.write_receipt(p,receipt)
    assert result["private_receipt_created"] is True
    assert p.stat().st_mode&0o077==0
    assert p.stat().st_uid==os.geteuid()
    assert json.loads(p.read_text())==receipt
    with pytest.raises(FileExistsError,match="overwrite"):
        transfer_receipt.write_receipt(p,receipt)
    assert verified(stage,receipt)["both_original_blinded_staging_bundles_verified"] is True


def test_bad_pinned_digest_or_human_claim_never_passes(stage):
    r=snapshot(stage)
    r["reviewer_1"]["files"]["batch_001.csv"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="receipt changed"):
        verified(stage,r)
    r=snapshot(stage)
    r["real_humans_independently_verified"]=True
    with pytest.raises(ValueError,match="authority claims"):
        verified(stage,r)


def test_modified_source_blinded_answer_invalidates_receipt(stage):
    receipt=snapshot(stage)
    f=stage[2]/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["model_response"]="altered hypothetical output"
    atomic_save(f,rows)
    with pytest.raises(ValueError):
        verified(stage,receipt)


def test_early_human_rating_invalidates_blank_source_handoff(stage):
    receipt=snapshot(stage)
    f=stage[2]/"batch_001.csv"
    rows=read_csv(f)
    rows[0].update(reviewer_id="synthetic",response_category="full_refusal",
                   actionable_harmful_content="no")
    atomic_save(f,rows)
    with pytest.raises(ValueError):
        verified(stage,receipt)


def test_unexpected_key_material_rejected_in_staging(stage):
    receipt=snapshot(stage)
    p=stage[3]/"private_model_key.json"
    p.write_text("{}")
    p.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected"):
        verified(stage,receipt)


def test_stage_hardlink_or_symlink_denied(stage):
    receipt=snapshot(stage)
    original=stage[2]/"batch_001.csv"
    alias=stage[4]/"hardlink"
    os.link(original,alias)
    with pytest.raises(ValueError,match="single-link"):
        verified(stage,receipt)
    alias.unlink()
    original.rename(stage[4]/"saved-original")
    original.symlink_to(stage[4]/"saved-original")
    with pytest.raises(ValueError,match="symlink"):
        verified(stage,receipt)


def test_swapped_original_reviewer_exports_rejected(stage):
    with pytest.raises(PermissionError,match="correctly keyed"):
        transfer_receipt.snapshot(stage[0],stage[1],stage[3],stage[2],
                                  source=source())


def test_missing_source_stage_file_is_not_a_valid_receipt(stage):
    receipt=snapshot(stage)
    (stage[3]/"batch_014.csv").unlink()
    with pytest.raises(ValueError,match="incomplete|unexpected|missing"):
        verified(stage,receipt)


@pytest.fixture
def copied(stage,monkeypatch):
    receipt=snapshot(stage)
    target1=stage[4].parent/"test-destination-user-1"
    target2=stage[4].parent/"test-destination-user-2"
    target1.mkdir(mode=0o700)
    target2.mkdir(mode=0o700)
    dest1=target1/"review"
    dest2=target2/"review"
    shutil.copytree(stage[2],dest1,copy_function=shutil.copy2)
    shutil.copytree(stage[3],dest2,copy_function=shutil.copy2)
    original=transfer_receipt._meta
    def synthetic_owners(path,directory=False):
        info=original(path,directory=directory)
        p=Path(path)
        if p==target1 or p==dest1 or p.parent==dest1:
            return SimpleNamespace(st_uid=10001,st_mode=info.st_mode,st_dev=info.st_dev,
                                   st_ino=info.st_ino,st_size=info.st_size,st_nlink=info.st_nlink)
        if p==target2 or p==dest2 or p.parent==dest2:
            return SimpleNamespace(st_uid=10002,st_mode=info.st_mode,st_dev=info.st_dev,
                                   st_ino=info.st_ino,st_size=info.st_size,st_nlink=info.st_nlink)
        return info
    monkeypatch.setattr(transfer_receipt,"_meta",synthetic_owners)
    # Original isolated.inspect consults original uid metadata, so the
    # mocked distinct Unix owners are ONLY synthetic contract fixtures.
    inspect_orig=isolated._check_private_path
    def mocked_inspection(path,directory=False):
        info=inspect_orig(path,directory=directory)
        p=Path(path)
        if p==dest1 or p.parent==dest1:
            return SimpleNamespace(st_uid=10001,st_mode=info.st_mode,
                                   st_size=info.st_size)
        if p==dest2 or p.parent==dest2:
            return SimpleNamespace(st_uid=10002,st_mode=info.st_mode,
                                   st_size=info.st_size)
        return info
    monkeypatch.setattr(isolated,"_check_private_path",mocked_inspection)
    monkeypatch.setattr(transfer_receipt.transfer,"_account",lambda uid:
        SimpleNamespace(pw_uid=uid,pw_shell="/bin/bash"))
    return stage,receipt,dest1,dest2


def valid_copy(copied,**kw):
    stage,receipt,d1,d2=copied
    opts=dict(first_target=d1,second_target=d2,first_uid=10001,
              second_uid=10002,source=source())
    opts.update(kw)
    return transfer_receipt.verify_transfer(
        receipt,*stage[:4],**opts)


def test_synthetic_transferred_bytes_and_per_file_owner_ledger(copied):
    result=valid_copy(copied)
    assert result["both_private_reviewer_transfers_byte_identical_to_staging"]
    assert result["distinct_existing_nonroot_reviewer_unix_owners"] is True
    assert result["model_key_transferred"] is False
    assert result["actual_human_personhood_or_consent_attested"] is False
    assert result["cross_reviewer_kernel_access_denials_verified"] is False
    assert result["reviewer_grading_authorized"] is False


def test_changed_transferred_file_rejected(copied):
    stage,receipt,d1,d2=copied
    f=d2/"batch_004.csv"
    rows=read_csv(f)
    rows[0]["model_response"]="altered in transit"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="do not match"):
        valid_copy(copied)


def test_ratings_added_before_transfer_verification_rejected(copied):
    stage,receipt,d1,d2=copied
    f=d1/"batch_006.csv"
    rows=read_csv(f)
    rows[0]["reviewer_id"]="fictional"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="do not match"):
        valid_copy(copied)


def test_model_key_added_to_transferred_workspace_rejected(copied):
    stage,receipt,d1,d2=copied
    extra=d1/"private_model_key.json"
    extra.write_text("{}")
    extra.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected"):
        valid_copy(copied)


def test_reused_key_uid_or_same_reviewer_uid_rejected(copied):
    stage,receipt,d1,d2=copied
    with pytest.raises(PermissionError,match="same Unix"):
        valid_copy(copied,second_uid=10001)
    with pytest.raises(PermissionError,match="model key custodian"):
        valid_copy(copied,first_uid=receipt["original_key_custodian_uid"])


def test_unknown_or_swapped_target_owner_rejected(copied):
    stage,receipt,d1,d2=copied
    with pytest.raises(PermissionError,match="parent UID"):
        valid_copy(copied,first_target=d2,second_target=d1)


def test_missing_transferred_file_rejected(copied):
    stage,receipt,d1,d2=copied
    (d1/"batch_015.csv").unlink()
    with pytest.raises(ValueError,match="unexpected"):
        valid_copy(copied)


def test_stage_changes_after_synthetic_transfer_rejected(copied):
    stage,receipt,d1,d2=copied
    f=stage[3]/"batch_010.csv"
    rows=read_csv(f)
    rows[0]["uncertainty_notes"]="synthetic write after transfer"
    atomic_save(f,rows)
    with pytest.raises(ValueError):
        valid_copy(copied)
