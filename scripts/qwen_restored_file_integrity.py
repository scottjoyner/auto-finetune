#!/usr/bin/env python3
"""Fail-closed read-only file integrity after a future isolated Qwen copy.

Compare exactly 22 files against a private trusted source manifest. This is
NOT a trainer deserialization/restore test, and NEVER authorizes a job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

from scripts import qwen_protected_backup_plan as plan_source

STEPS = (100, 150)
REQUIRED_FILE_COUNT = 22
MAX_MANIFEST_BYTES = 64 * 1024
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def _safe_file(path: Path) -> bool:
    try:
        s = path.lstat()
        return stat.S_ISREG(s.st_mode) and s.st_nlink == 1
    except OSError:
        return False


def _load_manifest(path: Path) -> tuple[dict | None, list[str]]:
    if not plan_source.all_directories_unlinked(path.parent) or not _safe_file(path):
        return None, ["trusted_manifest_file_unavailable_or_linked"]
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            return None, ["trusted_manifest_too_large"]
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None, ["trusted_manifest_invalid"]
    if not isinstance(doc, dict):
        return None, ["trusted_manifest_invalid"]
    reasons = []
    if (
        doc.get("expected_label") != plan_source.LABEL
        or doc.get("status") != "BACKUP_PLANNING_HOLD"
        or doc.get("read_only") is not True
        or doc.get("hashes_collected") is not True
        or doc.get("reasons") != []
        or doc.get("copy_started") is not False
        or doc.get("may_copy") is not False
    ):
        reasons.append("trusted_manifest_policy_invalid")
    protected = doc.get("protected_checkpoints")
    if not isinstance(protected, list) or len(protected) != 2:
        return None, ["trusted_manifest_checkpoint_count_invalid"]
    if [x.get("global_step") for x in protected if isinstance(x,dict)] != list(STEPS):
        reasons.append("trusted_manifest_step_mismatch")
    if doc.get("total_checkpoint_files") != REQUIRED_FILE_COUNT:
        reasons.append("trusted_manifest_file_count_mismatch")
    total = 0
    for row in protected:
        if not isinstance(row,dict):
            return None, ["trusted_manifest_checkpoint_row_invalid"]
        if row.get("checkpoint") != f"checkpoint-{row.get('global_step')}":
            reasons.append("trusted_manifest_step_mismatch")
        entries = row.get("files")
        if not isinstance(entries,list) or len(entries) != 11:
            reasons.append("trusted_manifest_checkpoint_files_invalid")
            continue
        names = []
        for entry in entries:
            if not isinstance(entry,dict):
                reasons.append("trusted_manifest_checkpoint_files_invalid")
                continue
            name, size, digest = (
                entry.get("filename"), entry.get("bytes"), entry.get("sha256")
            )
            if not isinstance(name,str) or not SAFE_NAME.fullmatch(name):
                reasons.append("trusted_manifest_filename_invalid")
            else:
                names.append(name)
            if type(size) is not int or size < 1:
                reasons.append("trusted_manifest_size_invalid")
            else:
                total += size
            if not isinstance(digest,str) or not HEX256.fullmatch(digest):
                reasons.append("trusted_manifest_checksum_invalid")
        if len(set(names)) != len(names):
            reasons.append("trusted_manifest_duplicate_name")
        if not plan_source.MANDATORY.issubset(names):
            reasons.append("trusted_manifest_mandatory_files_missing")
        if row.get("bytes") != sum(
            e.get("bytes",0) for e in entries
            if isinstance(e,dict) and type(e.get("bytes")) is int
        ):
            reasons.append("trusted_manifest_size_invalid")
    if doc.get("full_checkpoint_bytes") != total:
        reasons.append("trusted_manifest_total_size_mismatch")
    if total > plan_source.MAX_BYTES_TOTAL:
        reasons.append("trusted_manifest_total_size_exceeds_bound")
    return doc if not reasons else None, sorted(set(reasons))


def verify(manifest_path: Path, candidate_root: Path) -> dict:
    manifest, reasons = _load_manifest(manifest_path)
    rows = []
    complete = 0
    total_checked = 0
    total_bytes = 0

    trusted_root = plan_source.all_directories_unlinked(candidate_root)
    if not trusted_root:
        reasons.append("candidate_root_absent_or_symlinked")
    # Never score the production source tree as a successful restore.
    source_root = plan_source.a.SOURCE
    try:
        if candidate_root == source_root or source_root in candidate_root.parents:
            reasons.append("production_source_is_not_a_restore_candidate")
    except (OSError, ValueError):
        reasons.append("candidate_path_invalid")
    label = candidate_root / plan_source.LABEL
    if not plan_source.all_directories_unlinked(label):
        reasons.append("candidate_label_absent_or_symlinked")

    if manifest is not None and trusted_root and plan_source.all_directories_unlinked(label):
        for row in manifest["protected_checkpoints"]:
            candidate = label / row["checkpoint"]
            row_reasons = []
            matching = 0
            if not plan_source.all_directories_unlinked(candidate):
                row_reasons.append("candidate_checkpoint_absent_or_linked")
            else:
                try:
                    actual = sorted(candidate.iterdir())
                    expected = {entry["filename"] for entry in row["files"]}
                    if {path.name for path in actual} != expected:
                        row_reasons.append("candidate_extra_or_missing_files")
                except OSError:
                    actual = []
                    row_reasons.append("candidate_files_unreadable")
                if not row_reasons:
                    for entry in row["files"]:
                        path = candidate / entry["filename"]
                        if not _safe_file(path):
                            row_reasons.append("candidate_nonregular_or_linked_file")
                            continue
                        try:
                            before = path.stat()
                            if before.st_size != entry["bytes"]:
                                row_reasons.append("candidate_file_size_mismatch")
                                continue
                            digest=hashlib.sha256()
                            with path.open("rb") as f:
                                for chunk in iter(lambda:f.read(1024*1024),b""):
                                    digest.update(chunk)
                            after = path.stat()
                            if (
                                before.st_size, before.st_ino, before.st_dev,
                                before.st_mtime_ns, before.st_ctime_ns
                            ) != (
                                after.st_size, after.st_ino, after.st_dev,
                                after.st_mtime_ns, after.st_ctime_ns
                            ):
                                row_reasons.append("candidate_modified_during_hash")
                            elif digest.hexdigest() != entry["sha256"]:
                                row_reasons.append("candidate_checksum_mismatch")
                            else:
                                matching += 1
                                total_bytes += before.st_size
                        except OSError:
                            row_reasons.append("candidate_read_failure")
            total_checked += matching
            if not row_reasons and matching == len(row["files"]):
                complete += 1
            rows.append({
                "checkpoint":row["checkpoint"],
                "matching_files":matching,
                "expected_files":len(row["files"]),
                "problems":sorted(set(row_reasons)),
            })
    if complete != 2:
        reasons.append("full_file_integrity_unverified")

    return {
        "status":"NO_VERIFIED_RESTORE",
        "read_only":True,
        "full_file_manifest_trusted": manifest is not None,
        "candidate_file_count_matching":total_checked,
        "candidate_bytes_hash_matching":total_bytes,
        "complete_checkpoint_dirs_hash_matching":complete,
        "file_hash_match_only": manifest is not None and complete == 2 and not reasons,
        "restore_test_passed":False,
        "trainer_state_deserialized":False,
        "may_copy":False,
        "may_resume":False,
        "may_train":False,
        "may_prune":False,
        "may_release_source":False,
        "reasons":sorted(set(reasons)),
        "checkpoints":rows,
    }


def main() -> int:
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--trusted-manifest",type=Path,required=True)
    args.add_argument("--candidate-root",type=Path,required=True)
    opts=args.parse_args()
    print(json.dumps(verify(opts.trusted_manifest,opts.candidate_root),sort_keys=True))
    return 78


if __name__=="__main__":
    raise SystemExit(main())
