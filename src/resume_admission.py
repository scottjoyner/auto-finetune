"""Explicit Qwen3-8B checkpoint-resume admission. Read-only; no GPU work.

An adapter backup is NOT sufficient for a fully reproducible optimizer resume.
The live auditors intentionally return NO_GO until restore custody and a
separate restore trial are verified, so this feature is presently blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat

RUN = Path("/media/scott/data/finetune-staging/outputs/checkpoints/qwen3-8b-sft-r1")
STAGING = Path("/media/scott/data/finetune-staging")
EXPECTED_MODEL = STAGING / "models/Qwen3-8B"
SPLIT = STAGING / "data/future-runs/mixed-seed42/partition.json"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _isdir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _isfile(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def evaluate(requested: str | None, *, output_dir: str,
             data_path: str, model_name: str,
             backup: dict, preflight: dict,
             run_dir: Path = RUN, split: Path = SPLIT) -> Decision:
    """Check exact file identities and independent backup/restore policy.

    Neither flags nor environment variables can stand in for verified receipts.
    A return of allowed from a synthetic test is NOT production certification.
    """
    if not isinstance(requested, str) or not requested.startswith("/"):
        return Decision(False, "explicit_absolute_checkpoint_required")
    path = Path(requested)
    if not re.fullmatch(r"checkpoint-[1-9][0-9]*", path.name):
        return Decision(False, "invalid_numeric_checkpoint")
    # No resolution of paths into a different run, and no path traversal.
    if ".." in path.parts or path.parent != run_dir or Path(output_dir) != run_dir:
        return Decision(False, "checkpoint_output_mismatch")
    if not (_isdir(run_dir) and _isdir(path)):
        return Decision(False, "checkpoint_unavailable_or_symlinked")
    if Path(model_name) != EXPECTED_MODEL:
        return Decision(False, "base_model_mismatch")
    if not _isfile(split):
        return Decision(False, "split_manifest_absent")
    try:
        if split.stat().st_size > 2 * 1024 * 1024:
            return Decision(False, "split_manifest_invalid")
        partition = json.loads(split.read_text())
        split_source = partition["train_path"]
        contamination = partition["contamination"]
    except (OSError, KeyError, ValueError, TypeError, UnicodeError):
        return Decision(False, "split_manifest_invalid")
    if (not isinstance(split_source, str)
            or Path(data_path) != Path(split_source)
            or not _isfile(Path(data_path))):
        return Decision(False, "exact_training_split_mismatch")
    if not isinstance(contamination, dict) or (
            contamination.get("status") != "clean"
            or contamination.get("overlap_count") != 0):
        return Decision(False, "training_split_contaminated")
    if not isinstance(backup, dict) or not isinstance(preflight, dict):
        return Decision(False, "independent_audits_missing")
    if (backup.get("restore_test_passed") is not True
            or backup.get("status") != "VERIFIED_RESTORE"
            or backup.get("may_release_source") is not False):
        return Decision(False, "restore_custody_not_verified")
    if (preflight.get("restore_test_passed") is not True
            or preflight.get("may_resume") is not True
            or preflight.get("reasons") != []):
        return Decision(False, "resume_dependency_not_verified")
    checkpoint = preflight.get("checkpoint")
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("step") != int(path.name[11:])):
        return Decision(False, "verified_checkpoint_step_mismatch")
    return Decision(True, "independently_verified_resume")


def check_live(requested: str, *, output_dir: str, data_path: str, model_name: str) -> Decision:
    """Existing auditors do NOT grant restore permission. Always fail closed."""
    try:
        from scripts.resume_backup_audit import audit as audit_backup, SOURCE, BACKUPS
        from scripts.qwen_resume_preflight import audit as audit_dependencies
        backup = audit_backup(SOURCE, BACKUPS)
        preflight = audit_dependencies(STAGING, Path(__file__).resolve().parents[1],
                                       hash_partition_source=True)
        return evaluate(requested, output_dir=output_dir, data_path=data_path,
                        model_name=model_name, backup=backup, preflight=preflight)
    except Exception:
        return Decision(False, "independent_audits_unavailable")
