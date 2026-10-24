#!/usr/bin/env python3
"""Explicitly gated, non-destructive copy of the two protected Qwen checkpoints.

This NEVER runs a trainer, restores its optimizer, prunes a checkpoint,
updates a latest backup pointer or authorizes any further execution.
Default CLI mode is a read-only readiness report, exit78.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys

from scripts import qwen_protected_backup_plan as plan_source
from scripts import qwen_restored_file_integrity as integrity

SOURCE = plan_source.a.SOURCE
DESTINATION = plan_source.a.BACKUPS
STOP_FILE = Path("/media/scott/data/finetune-staging/launch-next.stop")
TRAIN_REPO = Path("/home/scott/git/auto-finetune")
MIN_FREE_MARGIN = 1 * 1024**3
EXPECTED_MOUNT = "/media/scott/SSD_4TB"
EXPECTED_NFS_SOURCE = "100.64.43.123:/media/scott/SSD_4TB"


def _expected_independent_nfs_mount(destination_parent: Path) -> bool:
    # st_dev comparison alone is insufficient if a network filesystem was
    # silently unmounted and the same local directory remains visible.
    if destination_parent != DESTINATION:
        return False
    try:
        for entry in Path("/proc/self/mountinfo").read_text().splitlines():
            left, sep, right = entry.partition(" - ")
            if not sep:
                continue
            columns, suffix = left.split(), right.split()
            if (len(columns) >= 6 and len(suffix) >= 2
                    and columns[4] == EXPECTED_MOUNT
                    and suffix[0] == "nfs4"
                    and suffix[1] == EXPECTED_NFS_SOURCE
                    and "rw" in columns[5].split(",")):
                return True
    except OSError:
        pass
    return False


class CustodyHold(RuntimeError):
    pass


def _hash_bytes(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda:fd.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _assert_hold(stop_file: Path, train_repo: Path, *, check_admission: bool) -> None:
    if not integrity._safe_file(stop_file):
        raise CustodyHold("training_stopfile_absent")
    if not check_admission:
        return
    try:
        r=subprocess.run([sys.executable,"-m","src.training_admission"],
                         cwd=train_repo, capture_output=True, timeout=12,
                         text=True, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise CustodyHold("training_admission_unavailable") from None
    if r.returncode!=78 or "HOLD" not in r.stdout:
        raise CustodyHold("training_admission_not_a_confirmed_hold")


def validate(
    manifest_path: Path, manifest_sha256: str, source: Path,
    destination_parent: Path, stop_file: Path, train_repo: Path,
    *, check_admission: bool = False,
) -> dict:
    reasons=[]
    if not isinstance(manifest_sha256,str) or not integrity.HEX256.fullmatch(manifest_sha256):
        reasons.append("source_manifest_digest_invalid")
    if not integrity._safe_file(manifest_path):
        reasons.append("source_manifest_unavailable")
    manifest,manifest_issues=integrity._load_manifest(manifest_path)
    reasons.extend(manifest_issues)
    if manifest and _hash_bytes(manifest_path)!=manifest_sha256:
        reasons.append("source_manifest_sha256_mismatch")

    if not plan_source.all_directories_unlinked(source):
        reasons.append("source_directory_untrusted")
    if not plan_source.all_directories_unlinked(destination_parent):
        reasons.append("destination_parent_untrusted")
    independent=False
    free=0
    try:
        independent=source.stat().st_dev != destination_parent.stat().st_dev
        free=os.statvfs(destination_parent)
        free=free.f_bavail*free.f_frsize
    except OSError:
        reasons.append("destination_mount_unavailable")
    if not independent:
        reasons.append("independent_backup_device_not_observed")
    if not _expected_independent_nfs_mount(destination_parent):
        reasons.append("pinned_nfs_backup_mount_not_verified")
    size=manifest.get("full_checkpoint_bytes",0) if manifest else 0
    if free < size*2+MIN_FREE_MARGIN:
        reasons.append("insufficient_non_destructive_backup_headroom")
    if not integrity._safe_file(stop_file):
        reasons.append("training_stopfile_absent")
    elif check_admission:
        try:
            _assert_hold(stop_file,train_repo,check_admission=True)
        except CustodyHold as err:
            reasons.append(str(err))

    if manifest is not None:
        clean=source/plan_source.LABEL
        if not plan_source.all_directories_unlinked(clean):
            reasons.append("source_label_untrusted")
        else:
            checkpoints=plan_source.a.checkpoint_rows(source)
            exact=[r for r in checkpoints
                   if r["label"]==plan_source.LABEL and r["protected_newest_two"]
                   and not r["quarantined"]]
            if [r["step"] for r in exact]!=[100,150]:
                reasons.append("protected_checkpoint_selection_changed")
            for row in manifest["protected_checkpoints"]:
                cp=clean/row["checkpoint"]
                if not plan_source.all_directories_unlinked(cp):
                    reasons.append("source_checkpoint_not_regular_directory")
                    continue
                try:
                    observed={p.name for p in cp.iterdir()}
                    expected={x["filename"] for x in row["files"]}
                    if observed!=expected:
                        reasons.append("source_checkpoint_file_list_changed")
                except OSError:
                    reasons.append("source_checkpoint_unreadable")
    return {
        "status":"CUSTODY_COPY_HOLD",
        "reasons":sorted(set(reasons)),
        "source_manifest_sha256":manifest_sha256,
        "source_manifest_eligible":not reasons and manifest is not None,
        "files":manifest.get("total_checkpoint_files",0) if manifest else 0,
        "expected_bytes":size,
        "observed_independent_device":independent,
        "free_destination_bytes":free,
        "copy_started":False,
        "copy_verified":False,
        "trainer_restore_passed":False,
        "may_train":False,
        "may_resume":False,
        "may_prune":False,
        "may_release_source":False,
    }


def _file_copy(source: Path, target: Path, *, expected: dict) -> None:
    fd=os.open(source, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
            raise CustodyHold("source_not_a_unique_regular_file")
        if s.st_size!=expected["bytes"]:
            raise CustodyHold("source_size_changed")
        outfd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        try:
            digest=hashlib.sha256()
            with os.fdopen(os.dup(fd),"rb",closefd=True) as inp, os.fdopen(os.dup(outfd),"wb",closefd=True) as out:
                while True:
                    chunk=inp.read(1024*1024)
                    if not chunk:break
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
            os.fsync(outfd)
        finally:
            os.close(outfd)
        after=os.fstat(fd)
        if (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev) != (
            after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev
        ):
            raise CustodyHold("source_changed_during_copy")
        if digest.hexdigest()!=expected["sha256"] or _hash_bytes(target)!=expected["sha256"]:
            raise CustodyHold("source_or_destination_hash_mismatch")
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, doc: dict) -> None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        data=(json.dumps(doc,sort_keys=True,indent=2)+"\n").encode()
        with os.fdopen(os.dup(fd),"wb") as writer:
            writer.write(data);writer.flush()
        os.fsync(fd)
    finally:os.close(fd)


def _write_pinned_manifest(
    trusted_path: Path, stage_path: Path, pinned_sha256: str, trusted_doc: dict,
) -> None:
    """Copy the exact approved manifest bytes, not a reserialized document.

    Rechecking the hash, policy payload and on-disk staging prevents the
    staged verifier from inadvertently using another manifest authority.
    """
    if not integrity._safe_file(trusted_path):
        raise CustodyHold("trusted_manifest_replaced_before_staging")
    try:
        payload = trusted_path.read_bytes()
    except OSError as exc:
        raise CustodyHold("trusted_manifest_unavailable_before_staging") from exc
    if (len(payload) > integrity.MAX_MANIFEST_BYTES
            or hashlib.sha256(payload).hexdigest() != pinned_sha256):
        raise CustodyHold("trusted_manifest_changed_during_copy")
    try:
        if json.loads(payload) != trusted_doc:
            raise CustodyHold("trusted_manifest_changed_during_copy")
    except (UnicodeError, ValueError, TypeError) as exc:
        raise CustodyHold("trusted_manifest_invalid_before_staging") from exc
    fd = os.open(
        stage_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(os.dup(fd), "wb") as writer:
            writer.write(payload)
            writer.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    if _hash_bytes(stage_path) != pinned_sha256:
        raise CustodyHold("staged_manifest_digest_mismatch")


def _sync_dir(path: Path) -> None:
    """Require directory metadata durability before publishing a backup."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise CustodyHold("backup_directory_fsync_failed") from exc
    finally:
        os.close(fd)


def _atomic_noreplace_publish(stage: Path, final: Path) -> None:
    """Linux renameat2(RENAME_NOREPLACE); refuse unsupported filesystems.

    An existence check followed by os.rename() could replace an empty
    concurrent generation. Never downgrade to standard os.rename().
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CustodyHold("atomic_noreplace_not_supported")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    AT_FDCWD = -100
    RENAME_NOREPLACE = 1
    result = renameat2(
        AT_FDCWD, os.fsencode(stage), AT_FDCWD, os.fsencode(final),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise CustodyHold("backup_generation_conflict")
    if error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP):
        raise CustodyHold("atomic_noreplace_not_supported")
    raise CustodyHold("atomic_noreplace_publish_failed")


def copy_snapshot(
    manifest_path: Path, manifest_sha256: str, source: Path,
    dest_parent: Path, stop_file: Path, train_repo: Path,
    *, generation: str | None = None,
) -> dict:
    verdict=validate(manifest_path,manifest_sha256,source,dest_parent,
                     stop_file,train_repo,check_admission=True)
    if verdict["reasons"]:
        raise CustodyHold("preflight_denied:"+",".join(verdict["reasons"]))
    manifest,_=integrity._load_manifest(manifest_path)
    assert manifest is not None
    if generation is None:
        generation=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not plan_source.a.STAMP.fullmatch(generation):
        raise CustodyHold("invalid_timestamp_generation")
    final=dest_parent/generation
    if final.exists() or final.is_symlink():
        raise CustodyHold("backup_generation_already_exists")

    token=secrets.token_hex(8)
    stage=dest_parent/(".pending-qwen-"+token)
    if stage.exists():
        raise CustodyHold("backup_stage_exists")
    # Explicit 0700 directories and 0600 O_EXCL file modes make a
    # process-global os.umask mutation unnecessary and prevent thread races.
    created=False
    try:
        stage.mkdir(mode=0o700)
        created=True
        checkpoint_root=stage/"resume-checkpoints"
        label=checkpoint_root/plan_source.LABEL
        checkpoint_root.mkdir(mode=0o700)
        label.mkdir(mode=0o700)
        for row in manifest["protected_checkpoints"]:
            _assert_hold(stop_file,train_repo,check_admission=True)
            dest=label/row["checkpoint"]
            dest.mkdir(mode=0o700)
            for entry in row["files"]:
                _assert_hold(stop_file,train_repo,check_admission=True)
                _file_copy(source/plan_source.LABEL/row["checkpoint"]/entry["filename"],
                           dest/entry["filename"],expected=entry)
        staged_manifest = stage / "source-manifest.json"
        _write_pinned_manifest(
            manifest_path, staged_manifest, manifest_sha256, manifest
        )
        # Always tie the independent verifier to the externally pinned
        # original digest. A hash match to an arbitrary staged manifest
        # alone cannot establish source-to-copy custody.
        if _hash_bytes(staged_manifest) != manifest_sha256:
            raise CustodyHold("staged_manifest_digest_mismatch")
        verification=integrity.verify(staged_manifest,checkpoint_root)
        if not verification["file_hash_match_only"] or verification["candidate_file_count_matching"]!=22:
            raise CustodyHold("staged_copy_did_not_match_all_22_hashes")
        if _hash_bytes(staged_manifest) != manifest_sha256:
            raise CustodyHold("staged_manifest_changed_during_verify")
        _write_json_exclusive(stage/"custody-copy-receipt.json",{
            "status":"FULL_FILE_HASH_MATCH_ONLY",
            "source_manifest_sha256":manifest_sha256,
            "full_files_verified":22,
            "full_bytes":verdict["expected_bytes"],
            "backup_generation":generation,
            "actual_trainer_restore_performed":False,
            "may_resume":False,
            "may_train":False,
            "may_prune":False,
            "may_release_source":False,
        })
        _assert_hold(stop_file,train_repo,check_admission=True)
        # Durably sync the content-bearing directories before publication,
        # refusing a backend that cannot acknowledge directory metadata.
        for row in manifest["protected_checkpoints"]:
            _sync_dir(label / row["checkpoint"])
        _sync_dir(label)
        _sync_dir(checkpoint_root)
        _sync_dir(stage)
        _assert_hold(stop_file,train_repo,check_admission=True)
        # Refuse overwrites even if a second writer creates the generation
        # after the earlier existence check.
        _atomic_noreplace_publish(stage, final)
        created=False
        # If this sync fails, the generation exists but is NOT certified as
        # durably published: fail closed and leave it for reconciliation.
        _sync_dir(dest_parent)
        return {
            "status":"FULL_FILE_HASH_MATCH_ONLY",
            "generation":generation,
            "copied_files":22,
            "copied_bytes":verdict["expected_bytes"],
            "trainer_restore_test_passed":False,
            "may_train":False,
            "may_resume":False,
            "may_prune":False,
            "may_release_source":False,
            "existing_latest_pointer_untouched":True,
        }
    finally:
        if created and stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest",type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--destination-parent",type=Path,default=DESTINATION)
    p.add_argument("--stopfile",type=Path,default=STOP_FILE)
    p.add_argument("--train-repo",type=Path,default=TRAIN_REPO)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--approval",default="")
    args=p.parse_args()
    if not args.execute:
        result=validate(args.source_manifest,args.manifest_sha256,args.source,
                        args.destination_parent,args.stopfile,args.train_repo,
                        check_admission=False)
        print(json.dumps(result,sort_keys=True))
        return 78
    # The production command is intentionally fixed to the known source,
    # destination and hold locations; the Python test API accepts disposable
    # paths, but an operator cannot redirect the CLI to arbitrary systems.
    if (args.source != SOURCE or args.destination_parent != DESTINATION
            or args.stopfile != STOP_FILE or args.train_repo != TRAIN_REPO):
        print(json.dumps({"status":"CUSTODY_COPY_HOLD",
                          "reason":"production_custody_paths_must_be_canonical"}))
        return 78
    expected_approval="APPROVE_NONDESTRUCTIVE_QWEN_CHECKPOINTS_100_150"
    if args.approval!=expected_approval:
        print(json.dumps({"status":"CUSTODY_COPY_HOLD",
                          "reason":"explicit_operator_approval_missing"}))
        return 78
    try:
        result=copy_snapshot(args.source_manifest,args.manifest_sha256,
                             args.source,args.destination_parent,
                             args.stopfile,args.train_repo)
    except (CustodyHold,OSError,ValueError) as err:
        print(json.dumps({"status":"CUSTODY_COPY_HOLD",
                          "reason":str(err)}))
        return 78
    print(json.dumps(result,sort_keys=True))
    # Still HOLD: full file hash match is not a trainer restore.
    return 78


if __name__=="__main__":
    raise SystemExit(main())
