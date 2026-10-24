#!/usr/bin/env python3
"""Read-only, full-file Qwen checkpoint custody manifest; NEVER starts backup.

The result is a candidate operator runbook, not authorization or a restore
receipt. No copy/write/mkdir/symlink traversal/model execution in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from scripts import resume_backup_audit as a

LABEL = "qwen3-8b-sft-r1"
EXPECTED_PROTECTED_STEPS = (100, 150)
MANDATORY = frozenset((*a.COMPONENTS, "adapter_config.json",
                       "adapter_model.safetensors", "tokenizer.json",
                       "tokenizer_config.json", "training_args.bin"))
MAX_FILES_PER_CHECKPOINT = 128
MAX_BYTES_TOTAL = 12 * 1024**3
HEADROOM_BYTES = 1 * 1024**3


def all_directories_unlinked(path: Path) -> bool:
    if not path.is_absolute():
        return False
    try:
        for parent in (path, *path.parents):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                return False
        return True
    except OSError:
        return False


def file_inventory(folder: Path, *, hash_files: bool = False) -> tuple[list[dict], list[str]]:
    problems = []
    result = []
    try:
        candidates = sorted(folder.iterdir())
    except OSError:
        return [], ["checkpoint_unreadable"]
    if len(candidates) > MAX_FILES_PER_CHECKPOINT:
        return [], ["too_many_checkpoint_entries"]
    for path in candidates:
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
                problems.append("nonregular_symlink_or_hardlinked_entry")
                continue
            if before.st_size < 1:
                problems.append("empty_checkpoint_file")
                continue
            item = {"filename":path.name, "bytes":before.st_size}
            if hash_files:
                item["sha256"] = a.stream_sha256(path)
                after = path.lstat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns,
                    before.st_ino, before.st_dev) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                    after.st_ino, after.st_dev
                ):
                    problems.append("source_mutated_during_hash")
            result.append(item)
        except (OSError, ValueError):
            problems.append("source_file_unreadable")
    return result, sorted(set(problems))


def plan(source: Path, backup_root: Path, *, hash_files: bool = False) -> dict:
    reasons=[]
    protected=[]
    if not all_directories_unlinked(source):
        reasons.append("source_root_absent_or_symlinked")
    if not all_directories_unlinked(backup_root):
        reasons.append("backup_root_absent_or_symlinked")
    label=source/LABEL
    if not all_directories_unlinked(label):
        reasons.append("clean_label_absent_or_symlinked")
    rows=[r for r in a.checkpoint_rows(source)
          if r["label"] == LABEL and r["protected_newest_two"] and
          not r["quarantined"]]
    # Pin exact custody checkpoints: if checkpoint-150 is a symlink,
    # missing, replaced or a new trainer checkpoint unexpectedly appears,
    # never silently substitute an older/newer pair.
    if tuple(r["step"] for r in rows) != EXPECTED_PROTECTED_STEPS:
        reasons.append("expected_protected_steps_not_verified")
    if len(rows)!=2:
        reasons.append("two_protected_clean_checkpoints_not_found")
    if all_directories_unlinked(label):
        try:
            for entry in label.iterdir():
                if a.CKPT.fullmatch(entry.name) and not a.directory(entry):
                    reasons.append("untrusted_checkpoint_entry")
        except OSError:
            reasons.append("cannot_list_clean_checkpoints")
    total_files=0
    total_bytes=0
    for row in rows:
        directory=label/row["checkpoint"]
        if not all_directories_unlinked(directory):
            reasons.append("protected_checkpoint_directory_untrusted")
            continue
        files, file_problems=file_inventory(directory,hash_files=hash_files)
        names={f["filename"] for f in files}
        missing=sorted(MANDATORY-names)
        if missing:
            reasons.append("required_full_checkpoint_files_missing")
        reasons.extend(file_problems)
        total_files+=len(files)
        total_bytes+=sum(f["bytes"] for f in files)
        protected.append({
            "checkpoint":row["checkpoint"],
            "global_step":row["step"],
            "files":files,
            "missing_mandatory":missing,
            "bytes":sum(f["bytes"] for f in files),
        })
    if total_bytes>MAX_BYTES_TOTAL:
        reasons.append("inventory_exceeds_bounded_size")
    try:
        src_dev=source.stat().st_dev
        dest_dev=backup_root.stat().st_dev
        independent=src_dev != dest_dev
        free=os.statvfs(backup_root)
        free_bytes=free.f_bavail*free.f_frsize
    except OSError:
        independent=False
        free_bytes=0
        reasons.append("cannot_stat_backup_mount")
    if not independent:
        reasons.append("backup_storage_independence_not_proven")
    if free_bytes < total_bytes*2+HEADROOM_BYTES:
        reasons.append("backup_capacity_headroom_unverified")
    if not hash_files:
        reasons.append("content_hashes_not_requested")
    return {
        "status":"BACKUP_PLANNING_HOLD",
        "read_only":True,
        "hashes_collected":hash_files,
        "expected_label":LABEL,
        "protected_checkpoints":protected,
        "protected_checkpoint_count":len(protected),
        "total_checkpoint_files":total_files,
        "full_checkpoint_bytes":total_bytes,
        "resume_component_bytes":sum(
            sum(r["component_sizes"].values()) for r in rows
        ),
        "backup_destination_is_independent_device":independent,
        "backup_free_bytes_reported":free_bytes,
        "headroom_requirement_bytes":total_bytes*2+HEADROOM_BYTES,
        "reasons":sorted(set(reasons)),
        "copy_started":False,
        "backup_verified":False,
        "restore_test_passed":False,
        "may_copy":False,
        "may_prune":False,
        "may_resume":False,
        "may_train":False,
        "may_release_source":False,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=a.SOURCE)
    parser.add_argument("--backup-root",type=Path,default=a.BACKUPS)
    parser.add_argument("--hash-files",action="store_true",
                        help="stream SHA256 for each protected checkpoint file; no writes")
    opts=parser.parse_args()
    print(json.dumps(plan(opts.source,opts.backup_root,hash_files=opts.hash_files),
                     sort_keys=True))
    return 78


if __name__=="__main__":
    raise SystemExit(main())
