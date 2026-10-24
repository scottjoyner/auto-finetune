"""No GPU or NAS required: compare disposable full-file copies to source manifest."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import qwen_protected_backup_plan as planner
from scripts import qwen_restored_file_integrity as v


def fixture(tmp_path):
    original=tmp_path/"source"
    backup=tmp_path/"nominal-backup"
    backup.mkdir()
    label=original/planner.LABEL
    label.mkdir(parents=True)
    for step in (100,150):
        cp=label/f"checkpoint-{step}"
        cp.mkdir()
        for name in sorted(planner.MANDATORY|{"README.md","chat_template.jinja"}):
            (cp/name).write_bytes((f"fixture-{step}-{name}").encode())
    d=planner.plan(original,backup,hash_files=True)
    assert d["total_checkpoint_files"]==22
    assert d["hashes_collected"] and d["reasons"]==[
        "backup_storage_independence_not_proven"
    ]
    # The trust envelope must be complete; fixture backup on same device
    # is independently expected NOT to qualify any real copy.
    d["reasons"]=[]
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps(d))
    manifest.chmod(0o600)
    restore=tmp_path/"restored-root"
    shutil.copytree(original,restore)
    return manifest,restore


def test_byte_identical_fixture_is_hash_match_only_not_restore_receipt(tmp_path):
    m,r=fixture(tmp_path)
    report=v.verify(m,r)
    assert report["status"]=="NO_VERIFIED_RESTORE"
    assert report["complete_checkpoint_dirs_hash_matching"]==2
    assert report["candidate_file_count_matching"]==22
    assert report["file_hash_match_only"]
    assert not report["restore_test_passed"]
    assert not report["trainer_state_deserialized"]
    assert not report["may_train"] and not report["may_resume"]
    assert not report["may_prune"]


def test_corrupted_equal_length_file_rejected(tmp_path):
    m,r=fixture(tmp_path)
    p=r/planner.LABEL/"checkpoint-150"/"optimizer.pt"
    p.write_bytes(b"X"*p.stat().st_size)
    report=v.verify(m,r)
    assert report["complete_checkpoint_dirs_hash_matching"]==1
    assert "candidate_checksum_mismatch" in report["checkpoints"][1]["problems"]
    assert not report["file_hash_match_only"]


def test_missing_adapter_rejected(tmp_path):
    m,r=fixture(tmp_path)
    (r/planner.LABEL/"checkpoint-100"/"adapter_model.safetensors").unlink()
    result=v.verify(m,r)
    assert "candidate_extra_or_missing_files" in result["checkpoints"][0]["problems"]


def test_unknown_extra_file_rejected(tmp_path):
    m,r=fixture(tmp_path)
    (r/planner.LABEL/"checkpoint-150"/"untracked.bin").write_bytes(b"unsafe")
    result=v.verify(m,r)
    assert "candidate_extra_or_missing_files" in result["checkpoints"][1]["problems"]


def test_symlink_substitution_rejected(tmp_path):
    m,r=fixture(tmp_path)
    p=r/planner.LABEL/"checkpoint-150"/"optimizer.pt"
    p.unlink()
    p.symlink_to(r/planner.LABEL/"checkpoint-100"/"optimizer.pt")
    result=v.verify(m,r)
    assert "candidate_nonregular_or_linked_file" in result["checkpoints"][1]["problems"]


def test_malicious_manifest_path_traversal_refused(tmp_path):
    m,r=fixture(tmp_path)
    doc=json.loads(m.read_text())
    doc["protected_checkpoints"][0]["files"][0]["filename"]="../x"
    m.write_text(json.dumps(doc))
    out=v.verify(m,r)
    assert not out["full_file_manifest_trusted"]
    assert "trusted_manifest_filename_invalid" in out["reasons"]


def test_manifest_missing_checksum_not_accepted(tmp_path):
    m,r=fixture(tmp_path)
    doc=json.loads(m.read_text())
    doc["protected_checkpoints"][1]["files"][0].pop("sha256")
    m.write_text(json.dumps(doc))
    out=v.verify(m,r)
    assert not out["full_file_manifest_trusted"]
    assert "trusted_manifest_checksum_invalid" in out["reasons"]


def test_manifest_marked_copy_executed_is_untrusted(tmp_path):
    m,r=fixture(tmp_path)
    doc=json.loads(m.read_text())
    doc["copy_started"]=True
    m.write_text(json.dumps(doc))
    out=v.verify(m,r)
    assert "trusted_manifest_policy_invalid" in out["reasons"]


def test_symlinked_manifest_file_denied(tmp_path):
    m,r=fixture(tmp_path)
    shadow=tmp_path/"alias.json"
    shadow.symlink_to(m)
    out=v.verify(shadow,r)
    assert "trusted_manifest_file_unavailable_or_linked" in out["reasons"]


def test_manifest_different_step_denied(tmp_path):
    m,r=fixture(tmp_path)
    doc=json.loads(m.read_text())
    doc["protected_checkpoints"][1]["global_step"]=175
    m.write_text(json.dumps(doc))
    out=v.verify(m,r)
    assert "trusted_manifest_step_mismatch" in out["reasons"]


def test_missing_candidate_root_denied(tmp_path):
    m,r=fixture(tmp_path)
    result=v.verify(m,tmp_path/"missing")
    assert "candidate_root_absent_or_symlinked" in result["reasons"]
    assert result["complete_checkpoint_dirs_hash_matching"]==0


def test_candidate_copy_directory_is_never_modified(tmp_path):
    m,r=fixture(tmp_path)
    before={str(x):x.stat().st_mtime_ns for x in r.rglob("*") if x.is_file()}
    v.verify(m,r)
    after={str(x):x.stat().st_mtime_ns for x in r.rglob("*") if x.is_file()}
    assert before==after


def test_cli_requires_explicit_args_and_always_exits_78(tmp_path,monkeypatch,capsys):
    m,r=fixture(tmp_path)
    monkeypatch.setattr("sys.argv",["qwen_restored_file_integrity.py",
        "--trusted-manifest",str(m),"--candidate-root",str(r)])
    assert v.main()==78
    payload=json.loads(capsys.readouterr().out)
    assert payload["file_hash_match_only"]
    assert payload["restore_test_passed"] is False


def test_trusted_manifest_with_symlinked_parent_refused(tmp_path):
    m,r=fixture(tmp_path)
    target=tmp_path/"manifests"
    target.mkdir()
    moved=target/"manifest.json"
    m.rename(moved)
    alias=tmp_path/"alias-manifests"
    alias.symlink_to(target,target_is_directory=True)
    out=v.verify(alias/"manifest.json",r)
    assert "trusted_manifest_file_unavailable_or_linked" in out["reasons"]
    assert not out["full_file_manifest_trusted"]
    assert not out["may_resume"]
