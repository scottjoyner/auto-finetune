"""Source and backup are synthetic local files; never write to live fleet."""
from pathlib import Path
import hashlib
import json
import os

import pytest

from scripts import qwen_protected_backup_plan as p


def fixture(tmp_path):
    source=tmp_path/"source"
    backup=tmp_path/"backup"
    backup.mkdir()
    root=source/p.LABEL
    root.mkdir(parents=True)
    for n in (50,100,150):
        cp=root/f"checkpoint-{n}"
        cp.mkdir()
        for filename in p.MANDATORY:
            (cp/filename).write_bytes((filename+":"+str(n)).encode())
        (cp/"README.md").write_text("synthetic doc")
        (cp/"chat_template.jinja").write_text("synthetic {{ tpl }}")
    return source,backup


def test_full_checkpoint_files_not_only_four_components(tmp_path):
    source,backup=fixture(tmp_path)
    d=p.plan(source,backup)
    assert d["status"]=="BACKUP_PLANNING_HOLD"
    assert d["protected_checkpoint_count"]==2
    assert [r["checkpoint"] for r in d["protected_checkpoints"]]==[
        "checkpoint-100","checkpoint-150"
    ]
    assert d["total_checkpoint_files"]==22
    assert d["full_checkpoint_bytes"]>d["resume_component_bytes"]
    assert not d["copy_started"] and not d["restore_test_passed"]
    assert not d["may_copy"] and not d["may_train"] and not d["may_resume"]


def test_hash_mode_matches_all_fixture_files(tmp_path):
    source,backup=fixture(tmp_path)
    d=p.plan(source,backup,hash_files=True)
    assert d["hashes_collected"]
    for row in d["protected_checkpoints"]:
        for entry in row["files"]:
            origin=source/p.LABEL/row["checkpoint"]/entry["filename"]
            assert entry["sha256"]==hashlib.sha256(origin.read_bytes()).hexdigest()
    assert "content_hashes_not_requested" not in d["reasons"]
    assert not d["backup_verified"]


def test_missing_adapter_disqualifies_full_checkpoint(tmp_path):
    source,backup=fixture(tmp_path)
    (source/p.LABEL/"checkpoint-150"/"adapter_model.safetensors").unlink()
    d=p.plan(source,backup)
    assert "required_full_checkpoint_files_missing" in d["reasons"]
    assert d["protected_checkpoints"][-1]["missing_mandatory"]==[
        "adapter_model.safetensors"
    ]


def test_symlink_and_special_entries_detected(tmp_path):
    source,backup=fixture(tmp_path)
    file=source/p.LABEL/"checkpoint-150"/"fake.json"
    file.symlink_to(source/p.LABEL/"checkpoint-100"/"README.md")
    d=p.plan(source,backup)
    assert "nonregular_symlink_or_hardlinked_entry" in d["reasons"]


def test_hardlinked_checkpoint_content_disqualified(tmp_path):
    source,backup=fixture(tmp_path)
    dest=source/p.LABEL/"checkpoint-150"/"linked.file"
    os.link(source/p.LABEL/"checkpoint-100"/"README.md",dest)
    d=p.plan(source,backup)
    assert "nonregular_symlink_or_hardlinked_entry" in d["reasons"]


def test_symlinked_checkpoint_root_disqualifies(tmp_path):
    source,backup=fixture(tmp_path)
    moved=tmp_path/"moved"
    (source/p.LABEL/"checkpoint-150").rename(moved)
    (source/p.LABEL/"checkpoint-150").symlink_to(moved,target_is_directory=True)
    d=p.plan(source,backup)
    assert "expected_protected_steps_not_verified" in d["reasons"]
    assert "untrusted_checkpoint_entry" in d["reasons"]
    assert d["protected_checkpoint_count"]==2


def test_symlinked_backup_root_disqualifies(tmp_path):
    source,backup=fixture(tmp_path)
    alias=tmp_path/"fake"
    alias.symlink_to(backup,target_is_directory=True)
    d=p.plan(source,alias)
    assert "backup_root_absent_or_symlinked" in d["reasons"]
    assert not d["may_copy"]


def test_missing_backup_root_denied(tmp_path):
    source,backup=fixture(tmp_path)
    d=p.plan(source,tmp_path/"no-such-backup")
    assert "backup_root_absent_or_symlinked" in d["reasons"]
    assert "cannot_stat_backup_mount" in d["reasons"]
    assert d["backup_free_bytes_reported"]==0


def test_same_device_does_not_prove_backup_is_independent(tmp_path):
    source,backup=fixture(tmp_path)
    d=p.plan(source,backup)
    assert "backup_storage_independence_not_proven" in d["reasons"]
    assert not d["backup_destination_is_independent_device"]


def test_mutating_while_hashing_disqualifies(monkeypatch,tmp_path):
    source,backup=fixture(tmp_path)
    original=p.a.stream_sha256
    seen=False
    def mutate(path):
        nonlocal seen
        value=original(path)
        if not seen:
            path.write_bytes(path.read_bytes()+b"x")
            seen=True
        return value
    monkeypatch.setattr(p.a,"stream_sha256",mutate)
    d=p.plan(source,backup,hash_files=True)
    assert "source_mutated_during_hash" in d["reasons"]


def test_does_not_touch_source_or_backup_content(tmp_path):
    source,backup=fixture(tmp_path)
    before={str(f):f.stat().st_mtime_ns for f in source.rglob("*") if f.is_file()}
    d=p.plan(source,backup,hash_files=True)
    after={str(f):f.stat().st_mtime_ns for f in source.rglob("*") if f.is_file()}
    assert before==after
    assert not list(backup.iterdir())
    assert not d["may_copy"]


def test_never_admits_even_under_synthetic_large_capacity(monkeypatch,tmp_path):
    source,backup=fixture(tmp_path)
    class FakeStat:
        f_bavail=1_000_000_000
        f_frsize=4096
    monkeypatch.setattr(p.os,"statvfs",lambda _:FakeStat())
    d=p.plan(source,backup,hash_files=True)
    assert d["backup_free_bytes_reported"]>1_000_000_000_000
    assert not d["may_copy"] and not d["may_prune"] and not d["may_release_source"]


def test_cli_always_returns_hold(monkeypatch,capsys):
    monkeypatch.setattr(p,"plan",lambda *a,**kw:{
        "status":"BACKUP_PLANNING_HOLD","may_copy":False,"may_train":False
    })
    monkeypatch.setattr("sys.argv",["qwen_protected_backup_plan.py"])
    assert p.main()==78
    result=json.loads(capsys.readouterr().out)
    assert result["status"]=="BACKUP_PLANNING_HOLD"


def test_aborted_contaminated_label_never_selected(tmp_path):
    source,backup=fixture(tmp_path)
    dirty=source/"qwen3-8b-sft-r1-ABORTED-contaminated"/"checkpoint-9999"
    dirty.mkdir(parents=True)
    (dirty/"optimizer.pt").write_text("contamination")
    d=p.plan(source,backup)
    assert all("ABORTED" not in r["checkpoint"] for r in d["protected_checkpoints"])
    assert d["protected_checkpoint_count"]==2
