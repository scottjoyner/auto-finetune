"""Disposable bytes only; no production Qwen or backend access."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from scripts import qwen_atomic_custody_copy as c
from scripts import qwen_protected_backup_plan as source_plan
from scripts import qwen_restored_file_integrity as v

GEN="20261023T223812Z"


@pytest.fixture(autouse=True)
def isolated_test_only_nfs_mount(monkeypatch):
    # Fixture uses private /dev/shm, distinct from the fixture source. The
    # installed CLI never accepts redirected production paths or mocks.
    monkeypatch.setattr(c, "_expected_independent_nfs_mount", lambda _: True)


@contextmanager
def inputs(tmp_path):
    source=tmp_path/"source"
    label=source/source_plan.LABEL
    label.mkdir(parents=True)
    for step in (50,100,150):
        cp=label/f"checkpoint-{step}"
        cp.mkdir()
        for name in sorted(source_plan.MANDATORY|{"README.md","chat_template.jinja"}):
            (cp/name).write_bytes(f"synthetic-{step}:{name}".encode())
    # /dev/shm has different st_dev from /tmp and is temporary/private.
    with tempfile.TemporaryDirectory(prefix="qwen-custody-test-",dir="/dev/shm") as deststr:
        dest=Path(deststr)
        manifest_doc=source_plan.plan(source,dest,hash_files=True)
        assert not manifest_doc["reasons"]
        assert manifest_doc["total_checkpoint_files"]==22
        manifest=tmp_path/"source-manifest.json"
        manifest.write_text(json.dumps(manifest_doc))
        manifest.chmod(0o600)
        checksum=hashlib.sha256(manifest.read_bytes()).hexdigest()
        stop=tmp_path/"training.stop"
        stop.write_text("STOP")
        repo=tmp_path/"mock-train-repo"
        module=repo/"src"
        module.mkdir(parents=True)
        (module/"__init__.py").write_text("")
        (module/"training_admission.py").write_text(
            'print("[train-admission] HOLD: stop_file");raise SystemExit(78)\n'
        )
        yield source,dest,manifest,checksum,stop,repo


def action(objects, **kw):
    source,dest,manifest,sha,stop,repo=objects
    return c.copy_snapshot(manifest,sha,source,dest,stop,repo,generation=GEN, **kw)


def test_non_destructive_copy_of_all_22_bytes_to_independent_fixture(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        (dest/"latest.txt").write_text("historical-generation-untouched")
        before={(x.relative_to(source).as_posix(),hashlib.sha256(x.read_bytes()).hexdigest())
                for x in source.rglob("*") if x.is_file()}
        result=action(objects)
        assert result["status"]=="FULL_FILE_HASH_MATCH_ONLY"
        assert result["copied_files"]==22
        assert result["copied_bytes"]>0
        assert result["trainer_restore_test_passed"] is False
        assert not result["may_train"] and not result["may_resume"]
        assert (dest/"latest.txt").read_text()=="historical-generation-untouched"
        generated=dest/GEN
        assert generated.is_dir()
        assert generated.stat().st_mode&0o077==0
        assert (generated/"custody-copy-receipt.json").is_file()
        verified=v.verify(generated/"source-manifest.json", generated/"resume-checkpoints")
        assert verified["file_hash_match_only"] is True
        assert verified["candidate_file_count_matching"]==22
        assert verified["restore_test_passed"] is False
        for p in generated.rglob("*"):
            assert not p.is_symlink()
            assert p.stat().st_mode&0o077==0
        after={(x.relative_to(source).as_posix(),hashlib.sha256(x.read_bytes()).hexdigest())
               for x in source.rglob("*") if x.is_file()}
        assert before==after


def test_preflight_requires_explicit_hold_while_no_copy(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        stop.unlink()
        report=c.validate(manifest,sha,source,dest,stop,repo,check_admission=True)
        assert "training_stopfile_absent" in report["reasons"]
        with pytest.raises(c.CustodyHold,match="training_stopfile_absent"):
            action(objects)
        assert not list(dest.iterdir())


def test_training_admission_not_hold_denied(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        (repo/"src/training_admission.py").write_text(
            'print("admitted");raise SystemExit(0)\n'
        )
        with pytest.raises(c.CustodyHold,match="training_admission_not_a_confirmed_hold"):
            action(objects)
        assert not list(dest.iterdir())


def test_single_snapshot_never_overwrites_existing_generation(tmp_path):
    with inputs(tmp_path) as objects:
        first=action(objects)
        with pytest.raises(c.CustodyHold,match="backup_generation_already_exists"):
            action(objects)
        assert (objects[1]/GEN/"custody-copy-receipt.json").is_file()


def test_bad_source_manifest_checksum_denied_before_copy(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        with pytest.raises(c.CustodyHold,match="source_manifest_sha256_mismatch"):
            c.copy_snapshot(manifest,"0"*64,source,dest,stop,repo,generation=GEN)
        assert not list(dest.iterdir())


def test_mutated_source_file_detected_and_private_stage_removed(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        p=source/source_plan.LABEL/"checkpoint-150"/"adapter_model.safetensors"
        p.write_bytes(b"spoofed"*10)
        with pytest.raises(c.CustodyHold,match="source_size_changed|source_or_destination_hash_mismatch"):
            action(objects)
        assert not list(dest.iterdir())


def test_missing_required_checkpoint_step_is_denied(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        (source/source_plan.LABEL/"checkpoint-150").rename(source/source_plan.LABEL/"checkpoint-175")
        with pytest.raises(c.CustodyHold,match="protected_checkpoint_selection_changed"):
            action(objects)
        assert not list(dest.iterdir())


def test_rejects_an_untrusted_symlink_source(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        cp=source/source_plan.LABEL/"checkpoint-150"
        cp.rename(source/source_plan.LABEL/"moved")
        cp.symlink_to(source/source_plan.LABEL/"moved",target_is_directory=True)
        with pytest.raises(c.CustodyHold,match="protected_checkpoint_selection_changed"):
            action(objects)
        assert not list(dest.iterdir())


def test_premature_revocation_of_stop_file_rolls_back_staging(tmp_path,monkeypatch):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        original=c._file_copy
        count=0
        def revoke(*args,**kwargs):
            nonlocal count
            original(*args,**kwargs)
            count+=1
            if count==1:
                stop.unlink()
        monkeypatch.setattr(c,"_file_copy",revoke)
        with pytest.raises(c.CustodyHold,match="training_stopfile_absent"):
            action(objects)
        assert not list(dest.iterdir())


def test_hash_mismatch_during_copy_leaves_no_snapshot(tmp_path,monkeypatch):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        original=c._file_copy
        count=0
        def fail_after(*args,**kwargs):
            nonlocal count
            original(*args,**kwargs)
            count+=1
            if count==5:
                raise c.CustodyHold("synthetic_transient_read_failure")
        monkeypatch.setattr(c,"_file_copy",fail_after)
        with pytest.raises(c.CustodyHold,match="synthetic_transient_read_failure"):
            action(objects)
        assert not list(dest.iterdir())


def test_symlinked_destination_root_rejected(tmp_path):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        alias=tmp_path/"backup-alias"
        alias.symlink_to(dest,target_is_directory=True)
        with pytest.raises(c.CustodyHold,match="destination_parent_untrusted"):
            c.copy_snapshot(manifest,sha,source,alias,stop,repo,generation=GEN)
        assert not list(dest.iterdir())


def test_read_only_default_cli_exits_78(monkeypatch,tmp_path,capsys):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        monkeypatch.setattr("sys.argv",[
            "qwen_atomic_custody_copy.py","--source-manifest",str(manifest),
            "--manifest-sha256",sha,"--source",str(source),
            "--destination-parent",str(dest),"--stopfile",str(stop),
            "--train-repo",str(repo)
        ])
        assert c.main()==78
        d=json.loads(capsys.readouterr().out)
        assert d["status"]=="CUSTODY_COPY_HOLD"
        assert d["copy_started"] is False
        assert not list(dest.iterdir())


def test_execute_cli_does_not_allow_fixture_or_redirected_paths(monkeypatch,tmp_path,capsys):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        monkeypatch.setattr("sys.argv",[
            "qwen_atomic_custody_copy.py","--source-manifest",str(manifest),
            "--manifest-sha256",sha,"--execute",
            "--approval","APPROVE_NONDESTRUCTIVE_QWEN_CHECKPOINTS_100_150",
            "--source",str(source),"--destination-parent",str(dest),
        ])
        assert c.main()==78
        d=json.loads(capsys.readouterr().out)
        assert d["reason"]=="production_custody_paths_must_be_canonical"
        assert not list(dest.iterdir())


def test_execute_cli_requires_approval(monkeypatch,capsys,tmp_path):
    with inputs(tmp_path) as objects:
        manifest=objects[2];sha=objects[3]
        monkeypatch.setattr("sys.argv",[
            "qwen_atomic_custody_copy.py","--source-manifest",str(manifest),
            "--manifest-sha256",sha,"--execute"
        ])
        assert c.main()==78
        d=json.loads(capsys.readouterr().out)
        assert d["reason"]=="explicit_operator_approval_missing"


def test_pinned_nfs_mount_must_be_verified_before_any_copy(tmp_path,monkeypatch):
    with inputs(tmp_path) as objects:
        source,dest,manifest,sha,stop,repo=objects
        monkeypatch.setattr(c,"_expected_independent_nfs_mount",lambda _: False)
        result=c.validate(manifest,sha,source,dest,stop,repo,check_admission=True)
        assert "pinned_nfs_backup_mount_not_verified" in result["reasons"]
        with pytest.raises(c.CustodyHold,match="pinned_nfs_backup_mount_not_verified"):
            action(objects)
        assert not list(dest.iterdir())
