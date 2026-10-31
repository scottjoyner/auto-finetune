"""Read-only local capacity preflight for substantial LFM3 research writes.

Does not free, reserve, move, delete, remount, or automatically launch jobs.
Small human-review and read-only audits do not require this compute gate.
"""
import argparse
import json
import math
import os
from pathlib import Path

GIB = 1024 ** 3
DEFAULT_FLOOR_GIB = 20
DEFAULT_RESERVE_GIB = 8


def assess_bytes(available, planned_write, reserve, floor):
    """Fail closed if anticipated write + reserve or minimum floor is unmet."""
    values = (available, planned_write, reserve, floor)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in values):
        raise ValueError("available/write/reserve/floor must be finite nonnegative numbers")
    required = max(floor, planned_write + reserve)
    return {
        "available_gib": round(available / GIB, 3),
        "planned_write_gib": round(planned_write / GIB, 3),
        "reserve_gib": round(reserve / GIB, 3),
        "minimum_floor_gib": round(floor / GIB, 3),
        "required_available_gib": round(required / GIB, 3),
        "large_write_allowed": available >= required,
    }


def stat_capacity(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError("target must be an existing directory")
    v = os.statvfs(path)
    available = v.f_bavail * v.f_frsize
    if available < 0:
        raise ValueError("invalid filesystem available-byte count")
    return available
def preflight(path, planned_write_gib=5, reserve_gib=DEFAULT_RESERVE_GIB,
              minimum_floor_gib=DEFAULT_FLOOR_GIB, nas_path=None):
    for n in (planned_write_gib,reserve_gib,minimum_floor_gib):
        if type(n) not in (int,float) or not math.isfinite(n) or n < 0:
            raise ValueError("nonnegative finite GiB limits required")
    target = Path(path)
    available = stat_capacity(target)
    decision = assess_bytes(available, planned_write_gib*GIB,
                            reserve_gib*GIB, minimum_floor_gib*GIB)
    decision["evidence_class"] = "read-only filesystem capacity preflight"
    decision["target"] = str(target)
    decision["filesystem_measurement"] = "statvfs.f_bavail times f_frsize"
    decision["writes_or_deletions_performed"] = False
    decision["space_reserved"] = False
    if nas_path is not None:
        nas = Path(nas_path)
        if not nas.is_mount():
            raise ValueError("NAS path not a separate mounted filesystem")
        if os.stat(target).st_dev == os.stat(nas).st_dev:
            raise ValueError("NAS must be a different filesystem from compute target")
        decision["nas_mounted_on_different_device"] = True
        decision["nas_available_gib"] = round(stat_capacity(nas)/GIB,3)
        decision["nas_migration_performed"] = False
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",type=Path,default=Path("/home/scott/git"))
    p.add_argument("--planned-write-gib",type=float,default=5)
    p.add_argument("--reserve-gib",type=float,default=DEFAULT_RESERVE_GIB)
    p.add_argument("--minimum-floor-gib",type=float,default=DEFAULT_FLOOR_GIB)
    p.add_argument("--nas",type=Path)
    args = p.parse_args()
    result = preflight(args.target,args.planned_write_gib,
                       args.reserve_gib,args.minimum_floor_gib,args.nas)
    print(json.dumps(result,indent=2))
    if not result["large_write_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
