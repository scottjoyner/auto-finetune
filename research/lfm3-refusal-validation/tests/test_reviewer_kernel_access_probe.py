"""Synthetic paths only; no real review contents or human identities used."""
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import reviewer_kernel_access_probe as access


def _private_file(path, *, mode=0o600):
    path.write_text("synthetic public harmless test string")
    path.chmod(mode)
    return path


def test_shared_owner_or_root_cannot_pass_three_account_preflight(tmp_path):
    packet=tmp_path/"packet"
    packet.mkdir(mode=0o700)
    _private_file(packet/"private_model_key.json")
    folders=[]
    for i in range(3):
        p=tmp_path/f"human-{i}"
        p.mkdir(mode=0o700)
        _private_file(p/("blinded_adjudication.csv" if i==2 else "batch_001.csv"))
        folders.append(p)
    with pytest.raises(PermissionError,match="three actual distinct"):
        access.preflight(packet,*folders)


def test_synthetic_metadata_checks_uniqueness_without_reading_contents(tmp_path,monkeypatch):
    packet=tmp_path/"packet"
    packet.mkdir(mode=0o700)
    key=_private_file(packet/"private_model_key.json")
    paths=[]
    for i in range(3):
        p=tmp_path/f"rater-{i}"
        p.mkdir(mode=0o700)
        _private_file(p/("blinded_adjudication.csv" if i==2 else "batch_001.csv"))
        paths.append(p)
    orig=access._private
    fake={str(key):10000}
    for i,p in enumerate(paths):
        fake[str(p)]=10001+i
        fake[str(p/("blinded_adjudication.csv" if i==2 else "batch_001.csv"))]=10001+i
    def three_owners(path,*,directory=False):
        data=orig(path,directory=directory)
        return SimpleNamespace(st_uid=fake.get(str(path),10000),
                               st_mode=data.st_mode,st_nlink=data.st_nlink)
    monkeypatch.setattr(access,"_private",three_owners)
    result=access.preflight(packet,*paths)
    assert result["three_distinct_non_root_reviewer_unix_owners"] is True
    assert result["kernel_denial_probes_executed"] is False
    assert result["real_independent_human_identities_attested"] is False
    assert result["model_release_authorized"] is False


def test_preflight_rejects_symlinks_world_access_and_wrong_owner(tmp_path,monkeypatch):
    packet=tmp_path/"packet"
    packet.mkdir(mode=0o700)
    _private_file(packet/"private_model_key.json")
    one=tmp_path/"one"
    one.mkdir(mode=0o700)
    _private_file(one/"batch_001.csv",mode=0o644)
    with pytest.raises(PermissionError,match="private to owner"):
        access._private(one/"batch_001.csv")
    (one/"batch_001.csv").chmod(0o600)
    alias=tmp_path/"alias"
    alias.symlink_to(one,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink ancestor"):
        access._private(alias/"batch_001.csv")
    link=tmp_path/"hard"
    os.link(one/"batch_001.csv",link)
    with pytest.raises(ValueError,match="single-link"):
        access._private(one/"batch_001.csv")


def test_unprivileged_probe_requires_matching_os_uids(monkeypatch):
    current=os.getuid()
    assert current!=0
    with pytest.raises(PermissionError,match="non-root"):
        access._require_unprivileged_uid(0)
    with pytest.raises(PermissionError,match="real/effective/saved"):
        access._require_unprivileged_uid(current+100000)
    assert access._require_unprivileged_uid(current) is True


def test_probe_reads_zero_file_bytes_on_real_synthetic_mode_denials(tmp_path,monkeypatch):
    uid=os.getuid()
    owned=_private_file(tmp_path/"own",mode=0o600)
    denied=[_private_file(tmp_path/f"blocked-{n}",mode=0o000) for n in range(3)]
    def never_read(*args,**kwargs):
        raise AssertionError("private review content may not be read by access probe")
    monkeypatch.setattr(access.os,"read",never_read)
    result=access.probe(1,uid,owned,*denied)
    assert result["own_private_review_file_opened"] is True
    assert result["both_other_reviewer_samples_open_denied"] is True
    assert result["model_key_open_denied"] is True
    assert result["file_contents_read"] is False
    assert result["admin_witnessed_preflight_and_process"] is False
    assert result["independent_real_person_attestation"] is False


def test_readable_foreign_file_is_hard_failure(tmp_path):
    uid=os.getuid()
    owned=_private_file(tmp_path/"own")
    foreign=_private_file(tmp_path/"foreign")
    other=_private_file(tmp_path/"other",mode=0o000)
    secret=_private_file(tmp_path/"secret",mode=0o000)
    with pytest.raises(PermissionError,match="was readable"):
        access.probe(1,uid,owned,foreign,other,secret)


def test_missing_or_symlinked_other_is_not_denial_proof(tmp_path):
    uid=os.getuid()
    owned=_private_file(tmp_path/"own")
    other=_private_file(tmp_path/"other",mode=0o000)
    secret=_private_file(tmp_path/"secret",mode=0o000)
    with pytest.raises(ValueError,match="ambiguous"):
        access.probe(1,uid,owned,tmp_path/"missing",other,secret)
    alias=tmp_path/"alias"
    alias.symlink_to(secret)
    with pytest.raises(ValueError,match="ambiguous"):
        access.probe(1,uid,owned,alias,other,secret)


def test_wrong_owner_or_shared_paths_refused(tmp_path):
    uid=os.getuid()
    owned=_private_file(tmp_path/"own")
    others=[_private_file(tmp_path/f"no-{i}",mode=0o000) for i in range(3)]
    with pytest.raises(ValueError,match="distinct"):
        access.probe(1,uid,owned,owned,others[1],others[2])
    with pytest.raises(PermissionError,match="UIDs differ"):
        access.probe(1,uid+1000,owned,*others)


def test_ambiguous_permission_error_is_not_misclassified(monkeypatch):
    def artificial(path,*,own_uid=None):
        if own_uid is None:
            raise FileNotFoundError(errno.ENOENT,"missing",str(path))
    monkeypatch.setattr(access,"_open_without_reading",artificial)
    with pytest.raises(ValueError,match="not proof"):
        access._require_open_denied("/synthetic/not-a-file")


def test_actual_uid_result_cannot_assert_human_personhood(tmp_path):
    uid=os.getuid()
    own=_private_file(tmp_path/"own")
    deny=[_private_file(tmp_path/f"deny-{i}",mode=0o000) for i in range(3)]
    result=access.probe(3,uid,own,*deny)
    serialized=json.dumps(result)
    assert "synthetic public harmless" not in serialized
    assert result["independent_real_person_attestation"] is False
    assert result["model_release_authorized"] is False


def test_two_original_account_preflight_without_third_human(tmp_path,monkeypatch):
    packet=tmp_path/"packet"
    packet.mkdir(mode=0o700)
    key=_private_file(packet/"private_model_key.json")
    dirs=[]
    for i in range(2):
        root=tmp_path/f"human-{i}"
        root.mkdir(mode=0o700)
        _private_file(root/"batch_001.csv")
        dirs.append(root)
    base=access._private
    def fake_owners(path,*,directory=False):
        st=base(path,directory=directory)
        p=Path(path)
        uid=10000 if p==packet or p==key else (
            10001 if p==dirs[0] or p.parent==dirs[0] else 10002)
        return SimpleNamespace(st_uid=uid,st_mode=st.st_mode,st_nlink=st.st_nlink)
    monkeypatch.setattr(access,"_private",fake_owners)
    result=access.preflight(packet,*dirs)
    assert result["two_distinct_non_root_reviewer_unix_owners"] is True
    assert result["third_adjudicator_account_present"] is False
    assert result["three_distinct_non_root_reviewer_unix_owners"] is False
    assert result["kernel_denial_probes_executed"] is False
    assert result["real_independent_human_identities_attested"] is False


def test_first_original_account_real_kernel_denial_of_other_and_key(tmp_path):
    uid=os.getuid()
    own=_private_file(tmp_path/"my-blinded-batch")
    other=_private_file(tmp_path/"other-private-batch",mode=0o000)
    key=_private_file(tmp_path/"private-key",mode=0o000)
    result=access.probe(1,uid,own,other,None,key)
    assert result["other_reviewer_samples_tested"]==1
    assert result["one_other_reviewer_sample_open_denied"] is True
    assert result["both_other_reviewer_samples_open_denied"] is False
    assert result["model_key_open_denied"] is True
    assert result["admin_witnessed_preflight_and_process"] is False


def test_third_account_cannot_skip_either_original_reviewer_denial(tmp_path):
    uid=os.getuid()
    own=_private_file(tmp_path/"my-blinded-batch")
    other=_private_file(tmp_path/"other-private-batch",mode=0o000)
    key=_private_file(tmp_path/"private-key",mode=0o000)
    with pytest.raises(ValueError,match="both original reviewer"):
        access.probe(3,uid,own,other,None,key)
