"""Read-only mount-aware recovery candidate classification for x1-370.

Do not infer that a large cache is on the exhausted filesystem merely
because it is under /home/scott/git. Never delete, move or reserve files.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from src.research_storage_preflight import GIB


def backing_device_matches(target_dev, candidate_dev):
    """Compare actual stat device IDs, not apparent pathname prefixes."""
    if (type(target_dev) is not int or type(candidate_dev) is not int
            or target_dev < 0 or candidate_dev < 0):
        raise ValueError("device numbers must be nonnegative integers")
    return target_dev == candidate_dev


def shortfall_gib(available_bytes, minimum_floor_gib=20):
    if (type(available_bytes) is not int or available_bytes < 0 or
            type(minimum_floor_gib) not in (int, float) or
            not math.isfinite(minimum_floor_gib) or minimum_floor_gib < 0):
        raise ValueError("invalid available bytes or floor")
    return round(max(0.0, minimum_floor_gib - available_bytes / GIB), 3)


def directory_components_include_symlink(path):
    path = Path(path).expanduser().absolute()
    return any(part.is_symlink() for part in (path, *path.parents))


def usage_bytes(path):
    """Optional du measurement, on the candidate's own filesystem only.

    du totals are never equivalent to bytes safely reclaimable: hardlinks,
    live writers, open-but-deleted files and other workloads affect df.
    """
    result = subprocess.run(
        ["du", "-s", "-B1", "-x", "--", str(path)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    field = result.stdout.split(None, 1)[0]
    result_bytes = int(field)
    if result_bytes < 0:
        raise ValueError("negative measured disk usage")
    return result_bytes


def inspect(target, candidates, floor_gib=20, include_du=False):
    target = Path(target).expanduser().absolute()
    if not target.is_dir() or directory_components_include_symlink(target):
        raise ValueError("target must be a real non-symlink directory")
    if not candidates:
        raise ValueError("at least one candidate path is required")
    target_dev = os.stat(target).st_dev
    fs = os.statvfs(target)
    available_bytes = fs.f_bavail * fs.f_frsize
    shortage = shortfall_gib(available_bytes, floor_gib)
    results = []
    for original in candidates:
        candidate = Path(original).expanduser().absolute()
        if not candidate.exists():
            raise ValueError("candidate does not exist: " + str(candidate))
        is_symlinked = directory_components_include_symlink(candidate)
        device = os.stat(candidate).st_dev
        same = backing_device_matches(target_dev, device)
        size = usage_bytes(candidate) if include_du and not is_symlinked else None
        results.append({
            "candidate": str(candidate),
            "candidate_st_dev": device,
            "on_target_filesystem": same,
            "symlink_component_detected": is_symlinked,
            "measured_disk_usage_gib": round(size / GIB, 3) if size is not None else None,
            "potentially_relevant_to_target_capacity": same and not is_symlinked,
            "automatic_cleanup_authorized": False,
        })
    return {
        "evidence_class": "read-only mount-aware disk recovery classification",
        "target_directory": str(target),
        "target_st_dev": target_dev,
        "target_unprivileged_available_gib": round(available_bytes / GIB, 3),
        "minimum_target_free_gib": floor_gib,
        "target_shortfall_gib": shortage,
        "candidate_count": len(results),
        "candidates": results,
        "estimated_bytes_safely_reclaimable": None,
        "cross_filesystem_bytes_can_free_target": False,
        "files_deleted_or_moved": False,
        "reserved_filesystem_settings_changed": False,
        "limitations": [
            "Device identity is checked with os.stat(...).st_dev, not path prefixes or global du",
            "Optional du measures usage on the candidate filesystem, not guaranteed df recovery",
            "Hard links, active jobs, open unlinked files and concurrent writers affect reclaim",
            "A same-device match does not prove a candidate is disposable or inactive",
            "This audit cannot clean caches or change ext4 reserved-block policies",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("/"))
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--minimum-floor-gib", type=float, default=20)
    parser.add_argument("--measure-usage", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect(args.target, args.candidate, args.minimum_floor_gib,
                             args.measure_usage), indent=2))


if __name__ == "__main__":
    main()
