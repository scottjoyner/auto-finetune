"""Synthetic transfer preflight: no human accounts or copy/chown performed."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import reviewer_transfer_preflight as transfer
from src import isolated_reviewer_workspace as isolated
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from tests.test_isolated_reviewer_workspace import originals
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def setup(originals,monkeypatch):
    packet,batches,storage=originals
    one,two=storage/"reviewer-one-staging",storage/"reviewer-two-staging"
    isolated.export(packet,batches,1,one,source=source())
    isolated.export(packet,batches,2,two,source=source())
    parent1=storage.parent/"reviewer-one-private-parent"
    parent2=storage.parent/"reviewer-two-private-parent"
    parent1.mkdir(mode=0o700)
    parent2.mkdir(mode=0o700)
    first,second=parent1/"reviewer-one",parent2/"reviewer-two"
    real=isolated._check_private_path
    def mock_owner(path, directory=False):
        result=real(path,directory=directory)
        if Path(path)==parent1:
            return SimpleNamespace(st_uid=10001,st_mode=result.st_mode,st_size=result.st_size)
        if Path(path)==parent2:
            return SimpleNamespace(st_uid=10002,st_mode=result.st_mode,st_size=result.st_size)
        return result
    monkeypatch.setattr(isolated,"_check_private_path",mock_owner)
    def named_account(uid):
        if uid not in (10001,10002):
            raise KeyError(uid)
        return SimpleNamespace(pw_uid=uid,pw_shell="/bin/bash")
    monkeypatch.setattr(transfer.pwd,"getpwuid",named_account)
    return packet,batches,one,two,first,second,storage


def run(a,**updates):
    packet,batches,one,two,first,second,storage=a
    kw=dict(packet=packet,canonical_batches=batches,first_stage=one,
            second_stage=two,first_destination=first,second_destination=second,
            first_uid=10001,second_uid=10002,source=source())
    kw.update(updates)
    return transfer.preflight(**kw)


def test_intact_dual_export_targets_pass_only_readiness_contract(setup):
    result=run(setup)
    assert result["both_key_free_staging_packets_source_verified"] is True
    assert result["original_study_verified_and_blank"] is True
    assert result["separate_private_account_owned_destination_parents"] is True
    assert result["staged_blinded_rows_each"]==1350
    assert result["reviewer_workspace_copy_completed"] is False
    assert result["reviewer_account_kernel_denials_verified"] is False
    assert result["human_identity_and_consent_independently_attested"] is False
    assert result["admin_approved_transfer"] is False
    assert result["model_release_authorized"] is False
    assert not setup[4].exists() and not setup[5].exists()


def test_shared_target_uid_and_source_key_owner_rejected(setup):
    with pytest.raises(PermissionError,match="share one Unix account"):
        run(setup,second_uid=10001)
    with pytest.raises(PermissionError,match="model-key UID"):
        run(setup,first_uid=os.getuid())


def test_missing_target_account_fails_closed(setup):
    with pytest.raises(PermissionError,match="does not exist"):
        run(setup,first_uid=12345678)


def test_modified_or_scored_staging_source_rejected(setup):
    f=setup[2]/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["reviewer_id"]="synthetic-human"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="pristine blank"):
        run(setup)


def test_modified_staged_model_response_rejected(setup):
    f=setup[2]/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["model_response"]="other synthetic model answer"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="source columns"):
        run(setup)


def test_staging_swap_of_original_reviewers_denied(setup):
    with pytest.raises(PermissionError,match="correctly keyed"):
        run(setup,first_stage=setup[3],second_stage=setup[2])


def test_existing_destination_never_overwritten(setup):
    setup[4].mkdir(mode=0o700)
    with pytest.raises(FileExistsError,match="must be absent"):
        run(setup)


def test_target_parent_cannot_be_inside_original_staging(setup):
    one=setup[2]
    new_parent=one/"nested"
    new_parent.mkdir(mode=0o700)
    with pytest.raises((ValueError,PermissionError),match="owner|overlap"):
        run(setup,first_destination=new_parent/"new-review")


def test_target_destination_symlink_ancestor_rejected(setup):
    alias=setup[6].parent/"alias-reviewer-private"
    alias.symlink_to(setup[4].parent,target_is_directory=True)
    with pytest.raises(ValueError,match="symlinked"):
        run(setup,first_destination=alias/"new")


def test_destination_parent_must_be_owner_only(setup):
    setup[4].parent.chmod(0o755)
    with pytest.raises(PermissionError,match="private"):
        run(setup)


def test_destination_space_shortfall_rejected(setup,monkeypatch):
    monkeypatch.setattr(transfer.os,"statvfs",lambda _:SimpleNamespace(
        f_bavail=0,f_frsize=4096))
    with pytest.raises(OSError,match="free-space"):
        run(setup)


def test_partial_canonical_review_rejected_without_export_or_copy(setup):
    f=setup[1]/"reviewer_2"/"batch_001.csv"
    rows=read_csv(f)
    rows[0].update(reviewer_id="synthetic",response_category="full_refusal",
                   actionable_harmful_content="no")
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="fully blank"):
        run(setup)
    assert not setup[4].exists() and not setup[5].exists()


def test_noninteractive_os_account_not_acceptable(setup,monkeypatch):
    monkeypatch.setattr(transfer.pwd,"getpwuid",lambda uid:
        SimpleNamespace(pw_uid=uid,pw_shell="/usr/sbin/nologin"))
    with pytest.raises(PermissionError,match="not interactive"):
        run(setup)
