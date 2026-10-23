#!/usr/bin/env python3
"""Read-only evidence of checkpoint resume COMPONENT coverage, never a restore receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

SOURCE = Path("/media/scott/data/finetune-staging/outputs/checkpoints")
BACKUPS = Path("/media/scott/SSD_4TB/agent-state-backups/ml-state-backups/xwing")
COMPONENTS = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
STAMP = re.compile(r"[0-9]{8}T[0-9]{6}Z\Z")
CKPT = re.compile(r"checkpoint-([0-9]+)\Z")
LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\Z")


def regular(path: Path) -> bool:
    """Reject symlinks, directories, specials and unreadable/missing files."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generations(backups: Path) -> list[str]:
    if not directory(backups):
        return []
    return sorted(
        p.name for p in backups.iterdir()
        if STAMP.fullmatch(p.name) and directory(p)
    )


def choose_generation(backups: Path, available: list[str]) -> tuple[str | None, str]:
    pointer = backups / "latest.txt"
    if regular(pointer):
        try:
            stamp = pointer.read_text(encoding="utf-8").strip()
            if STAMP.fullmatch(stamp) and stamp in available:
                return stamp, "latest_pointer"
        except (OSError, UnicodeError):
            pass
    # A stale or forged latest pointer is NOT evidence of a valid generation.
    return None, "latest_pointer_unverified"


def checkpoint_rows(source: Path) -> list[dict]:
    if not directory(source):
        return []
    rows: list[dict] = []
    for label in sorted(source.iterdir()):
        if not directory(label) or not LABEL.fullmatch(label.name):
            continue
        checkpoints = []
        for ckpt in label.iterdir():
            matched = CKPT.fullmatch(ckpt.name)
            if matched and directory(ckpt):
                checkpoints.append((int(matched.group(1)), ckpt))
        checkpoints.sort(key=lambda pair: pair[0])
        quarantined = "aborted" in label.name.lower() or "contaminated" in label.name.lower()
        for rank, (step, ckpt) in enumerate(checkpoints):
            files = {}
            for component in COMPONENTS:
                path = ckpt / component
                if regular(path):
                    try:
                        files[component] = path.stat().st_size
                    except OSError:
                        continue
            rows.append({
                "label": label.name,
                "checkpoint": ckpt.name,
                "step": step,
                "quarantined": quarantined,
                "protected_newest_two": rank >= len(checkpoints) - 2,
                "component_sizes": files,
            })
    return rows


def verify_row(source: Path, backups: Path, generation: str | None, row: dict) -> dict:
    """Hash only where both candidate files exist; missing backups cost no I/O."""
    result = {k: row[k] for k in
              ("label", "checkpoint", "step", "quarantined", "protected_newest_two")}
    present = row["component_sizes"]
    result["source_components"] = sorted(present)
    result["source_bytes"] = sum(present.values())
    result["backup_matching_components"] = []
    result["missing_or_unverified_components"] = []
    result["component_mismatch"] = []
    if generation is None:
        result["missing_or_unverified_components"] = list(COMPONENTS)
        result["status"] = "BACKUP_GENERATION_UNVERIFIED"
        return result
    for component in COMPONENTS:
        if component not in present:
            result["missing_or_unverified_components"].append(component)
            continue
        src = source / row["label"] / row["checkpoint"] / component
        target = backups / generation / "resume-checkpoints" / row["label"] / row["checkpoint"] / component
        parents = (backups / generation / "resume-checkpoints",
                   backups / generation / "resume-checkpoints" / row["label"],
                   backups / generation / "resume-checkpoints" / row["label"] / row["checkpoint"])
        if not all(directory(p) for p in parents) or not regular(target):
            result["missing_or_unverified_components"].append(component)
            continue
        try:
            if target.stat().st_size != present[component]:
                result["component_mismatch"].append(component)
            elif stream_sha256(src) == stream_sha256(target):
                result["backup_matching_components"].append(component)
            else:
                result["component_mismatch"].append(component)
        except OSError:
            result["missing_or_unverified_components"].append(component)
    if result["component_mismatch"]:
        result["status"] = "COMPONENT_HASH_MISMATCH"
    elif result["missing_or_unverified_components"]:
        result["status"] = "INCOMPLETE_COMPONENT_BACKUP"
    else:
        result["status"] = "COMPONENT_HASH_MATCH_ONLY"
    return result


def audit(source: Path, backups: Path) -> dict:
    available = generations(backups)
    generation, pointer_state = choose_generation(backups, available)
    rows = checkpoint_rows(source)
    verified = [verify_row(source, backups, generation, r) for r in rows]
    protected = [r for r in verified if r["protected_newest_two"] and not r["quarantined"]]
    complete = sum(r["status"] == "COMPONENT_HASH_MATCH_ONLY" for r in protected)
    return {
        "status": "NO_VERIFIED_RESTORE",
        "read_only": True,
        "may_prune": False,
        "may_release_source": False,
        "restore_test_passed": False,
        "source_available": directory(source),
        "backup_available": directory(backups),
        "backup_generations": available,
        "latest_pointer_state": pointer_state,
        "checked_generation": generation,
        "checkpoint_dirs": len(verified),
        "quarantined_dirs": sum(r["quarantined"] for r in verified),
        "protected_production_dirs": len(protected),
        "protected_production_resume_bytes": sum(r["source_bytes"] for r in protected),
        "protected_component_hash_matches": complete,
        "protected_missing_or_mismatched": len(protected) - complete,
        "all_resume_component_bytes": sum(r["source_bytes"] for r in verified),
        "entries": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--backup-root", type=Path, default=BACKUPS)
    args = parser.parse_args()
    report = audit(args.source, args.backup_root)
    print(json.dumps(report, sort_keys=True))
    # A component hash match alone is not a successful restore drill.
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
