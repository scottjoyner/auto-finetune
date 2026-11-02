"""Synthetic-only metadata/namespace binding tests; not human attestation."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from src import reviewer_kernel_binding as bind
from src import reviewer_kernel_access_probe as kernel


def _file(p,mode=0o600):
    p.write_text("harmless synthetic probe target")
    p.chmod(mode)
    return p


@pytest.fixture
def simulated_accounts(tmp_path,monkeypatch):
    uid=os.getuid()
    packet=tmp_path/"packet"
    packet.mkdir(mode=0o700)
    key=_file(packet/"private_model_key.json",0o000)
    dirs=[]
    paths={}
    for i in (1,2,3):
        d=tmp_path/f"role-{i}"
        d.mkdir(mode=0o700)
        f=_file(d/("blinded_adjudication.csv" if i==3 else "batch_001.csv"),
                0o600 if i==1 else 0o000)
        dirs.append(d)
        paths[i]=f
    owners={str(packet):uid+7000,str(key):uid+7000}
    for i,d in enumerate(dirs,1):
        owners[str(d)]=uid if i==1 else uid+i*1000
        owners[str(paths[i])]=uid if i==1 else uid+i*1000
    real=kernel._private
    def pretend_metadata(path,*,directory=False):
        st=real(path,directory=directory)
        # This UID substitution proves only contract validation, NOT real
        # separate Unix owners or human account isolation.
        return SimpleNamespace(st_uid=owners[str(path)],st_mode=st.st_mode,
            st_dev=st.st_dev,st_ino=st.st_ino,st_nlink=st.st_nlink)
    monkeypatch.setattr(kernel,"_private",pretend_metadata)
    return uid,packet,dirs,paths,key


def test_preflight_binds_owner_and_inode_without_disclosing_paths(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    state=bind._validated_binding(receipt)
    assert state["files"]["reviewer_1"]["st_uid"]==uid
    assert set(state["files"])=={"model_key","reviewer_1","reviewer_2"}
    assert state["human_identity_attested"] is False
    assert state["administrator_witnessed_probes"] is False
    assert "role-1" not in json.dumps(receipt)
    assert str(packet) not in json.dumps(receipt)
    assert len(receipt["sha256"])==64


def test_bound_role_one_performs_real_synthetic_kernel_open_checks(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    result=bind.bound_probe(receipt,1,uid,paths[1],paths[2],None,key)
    assert result["all_probe_path_hashes_match_preflight"] is True
    assert result["actual_opened_own_file_matches_preflight_inode"] is True
    assert result["same_boot_and_linux_mount_user_namespaces"] is True
    assert result["original_reviewer_denial_count"]==1
    assert result["forbidden_target_inodes_same_during_probe_proved"] is False
    assert result["administrator_witnessed_process"] is False
    assert result["model_release_authorized"] is False
    result2=bind.postflight(receipt,packet,dirs[0],dirs[1])
    assert result2["preflight_postflight_metadata_same"] is True
    assert result2["independent_human_identity_attested"] is False


def test_preflight_binding_rejects_wrong_roster_path_and_expected_uid(simulated_accounts,tmp_path):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    wrong=_file(tmp_path/"wrong",0o000)
    with pytest.raises(ValueError,match="paths do not match"):
        bind.bound_probe(receipt,1,uid,paths[1],wrong,None,key)
    with pytest.raises(PermissionError,match="runtime reviewer UID"):
        bind.bound_probe(receipt,1,uid+1,paths[1],paths[2],None,key)
    with pytest.raises(ValueError,match="roster"):
        bind.bound_probe(receipt,1,uid,paths[1],paths[2],wrong,key)


def test_own_file_replacement_after_preflight_is_detected(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    own=paths[1]
    own.unlink()
    _file(own)
    with pytest.raises(PermissionError,match="own opened file differs"):
        bind.bound_probe(receipt,1,uid,own,paths[2],None,key)
    with pytest.raises(ValueError,match="metadata or host context drifted"):
        bind.postflight(receipt,packet,dirs[0],dirs[1])


def test_forbidden_file_swap_is_caught_by_custodian_postflight(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    other=paths[2]
    other.unlink()
    _file(other,0o000)
    # The unprivileged negative open still returns permission-denied.
    # Only a second privileged metadata snapshot detects inode replacement.
    result=bind.bound_probe(receipt,1,uid,paths[1],other,None,key)
    assert result["forbidden_target_inodes_same_during_probe_proved"] is False
    with pytest.raises(ValueError,match="metadata or host context drifted"):
        bind.postflight(receipt,packet,dirs[0],dirs[1])


def test_namespace_change_cannot_claim_same_context(simulated_accounts,monkeypatch):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    original=bind._machine
    def other_namespace():
        state=original()
        return dict(state,mount_namespace="mnt:[999]")
    monkeypatch.setattr(bind,"_machine",other_namespace)
    with pytest.raises(ValueError,match="namespace differs"):
        bind.bound_probe(receipt,1,uid,paths[1],paths[2],None,key)


def test_forged_or_relabeled_receipt_fails_by_default(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    forged=json.loads(json.dumps(receipt))
    forged["preflight"]["human_identity_attested"]=True
    with pytest.raises(ValueError,match="binding"):
        bind.bound_probe(forged,1,uid,paths[1],paths[2],None,key)
    forged=json.loads(json.dumps(receipt))
    forged["preflight"]["files"]["reviewer_1"]["st_ino"]+=1
    with pytest.raises(ValueError,match="binding"):
        bind.bound_probe(forged,1,uid,paths[1],paths[2],None,key)
    assert receipt["preflight"]["human_identity_attested"] is False


def test_three_person_roster_requires_both_other_paths_even_if_role_one(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1],dirs[2])
    assert set(receipt["preflight"]["files"])=={
        "model_key","reviewer_1","reviewer_2","reviewer_3"}
    with pytest.raises(ValueError,match="complete original reviewer roster"):
        bind.bound_probe(receipt,1,uid,paths[1],paths[2],None,key)
    result=bind.bound_probe(receipt,1,uid,paths[1],paths[2],paths[3],key)
    assert result["original_reviewer_denial_count"]==2
    assert result["administrator_witnessed_process"] is False


def test_preflight_refuses_relative_paths(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    with pytest.raises(ValueError,match="absolute paths"):
        bind.preflight(packet,Path("relative-reviewer-dir"),dirs[1])


def test_receipt_digest_is_not_digital_signature(simulated_accounts):
    uid,packet,dirs,paths,key=simulated_accounts
    receipt=bind.preflight(packet,dirs[0],dirs[1])
    p=json.loads(json.dumps(receipt))
    # A user with write access could rehash a local receipt; never
    # represent the SHA-256 check as trusted administrator attestation.
    p["preflight"]["files"]["reviewer_2"]["st_uid"]+=1
    p["sha256"]=bind._digest(p["preflight"])
    assert bind._validated_binding(p)["administrator_witnessed_probes"] is False
    with pytest.raises(ValueError,match="metadata or host context drifted"):
        bind.postflight(p,packet,dirs[0],dirs[1])
