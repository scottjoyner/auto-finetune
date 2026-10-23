#!/usr/bin/env python3
"""Read-only historic Qwen resume-component coverage: never a restore certificate.

Use the existing, fail-closed verifier for exact component matching. Each
backup generation is independently complete or incomplete: components from
different timestamps must NEVER be mixed into a synthetic "full" backup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import resume_backup_audit as a

MAX_GENERATIONS = 32


def audit_history(source: Path, backup_root: Path, *, max_generations: int = 12) -> dict:
    if type(max_generations) is not int or not 1 <= max_generations <= MAX_GENERATIONS:
        raise ValueError("max_generations must be an integer between 1 and 32")
    available = a.generations(backup_root)
    checked = list(reversed(available[-max_generations:]))
    rows = a.checkpoint_rows(source)
    protected = [
        r for r in rows
        if r["protected_newest_two"] and not r["quarantined"]
    ]
    observations: list[dict] = []
    for stamp in checked:
        matches = 0
        mismatches = 0
        missing = 0
        checked_bytes = 0
        for row in protected:
            outcome = a.verify_row(source, backup_root, stamp, row)
            if outcome["status"] == "COMPONENT_HASH_MATCH_ONLY":
                matches += 1
                checked_bytes += outcome["source_bytes"]
            elif outcome["status"] == "COMPONENT_HASH_MISMATCH":
                mismatches += 1
            else:
                missing += 1
        observations.append({
            "backup_generation": stamp,
            "protected_component_sets_hash_matching": matches,
            "protected_component_sets_incomplete": missing,
            "protected_component_sets_mismatched": mismatches,
            "protected_component_sets_complete": len(protected) > 0 and matches == len(protected),
            "component_bytes_matching": checked_bytes,
            "restore_attempt_observed": False,
            "restore_test_passed": False,
        })
    # Coverage and source evidence are deliberately not a restoration test.
    # No generation, even if fully component-matched, is authorization to train,
    # delete checkpoints, retire source media or declare a safe cutover.
    return {
        "status": "NO_VERIFIED_RESTORE",
        "read_only": True,
        "source_available": a.directory(source),
        "backup_available": a.directory(backup_root),
        "generations_found": len(available),
        "generations_checked": len(checked),
        "scan_truncated_by_limit": len(checked) < len(available),
        "protected_production_checkpoint_dirs": len(protected),
        "protected_production_resume_bytes": sum(
            sum(r["component_sizes"].values()) for r in protected
        ),
        "any_single_generation_hash_covers_all_protected": any(
            row["protected_component_sets_complete"] for row in observations
        ),
        "restore_test_passed": False,
        "may_copy": False,
        "may_prune": False,
        "may_resume": False,
        "may_train": False,
        "may_release_source": False,
        "generations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=a.SOURCE)
    parser.add_argument("--backup-root", type=Path, default=a.BACKUPS)
    parser.add_argument("--max-generations", type=int, default=12)
    args = parser.parse_args()
    try:
        result = audit_history(
            args.source, args.backup_root, max_generations=args.max_generations
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
