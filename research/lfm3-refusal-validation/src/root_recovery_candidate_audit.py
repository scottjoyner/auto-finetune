"""Read-only root-capacity recovery candidate audit for LFM3.

Do not confuse file age or lsof silence with safe disposition. This report
NEVER archives, copies, moves, truncates, deletes, chmods, or changes ext4
reserved blocks. Unprivileged statvfs.f_bavail controls root write safety.
Candidate st_blocks are ONLY an upper-bound estimate of what retiring a file
could return; open handles, consumers, hardlinks, filesystem metadata and
concurrent writers can reduce or prevent recovery.
"""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import time

from src.research_storage_preflight import GIB, preflight


def _real_path(path):
    p=Path(path).expanduser().absolute()
    if any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError("recovery candidate has a symlinked component")
    return p


def candidate(path,root_dev):
    p=_real_path(path)
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
        raise ValueError("recovery candidate must be a single-link regular file")
    if s.st_dev!=root_dev:
        raise ValueError("candidate cannot reclaim bytes from a different filesystem")
    blocks=s.st_blocks*512
    if blocks<0 or s.st_size<0:
        raise ValueError("invalid candidate disk allocation metadata")
    return {
        "path":str(p),
        "st_dev":s.st_dev,
        "st_ino":s.st_ino,
        "owner_uid":s.st_uid,
        "logical_bytes":s.st_size,
        "allocated_bytes_upper_bound":blocks,
        "mtime_ns":s.st_mtime_ns,
        "open_handles_independently_excluded":False,
        "future_consumers_excluded":False,
        "integrity_checksum_verified":False,
        "approved_for_move_or_removal":False,
    }


def calculate(samples,entries,*,floor_gib=20,planned_write_gib=5,
              reserve_gib=8,max_decline_gib=0.25):
    """Classify capacity from observations; never grant destructive action."""
    for name,value in (("floor_gib",floor_gib),("planned_write_gib",planned_write_gib),
                       ("reserve_gib",reserve_gib),("max_decline_gib",max_decline_gib)):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError("invalid nonnegative finite capacity policy: "+name)
    if len(samples)<2:
        raise ValueError("at least two capacity observations required")
    for s in samples:
        if (set(s)!={"monotonic_ns","available_bytes","device"} or
            any(type(s[k]) is not int or s[k]<0 for k in s)):
            raise ValueError("invalid capacity sample")
    if any(x["monotonic_ns"]>=y["monotonic_ns"] or x["device"]!=y["device"]
           for x,y in zip(samples,samples[1:])):
        raise ValueError("capacity samples not chronological on identical device")
    if any(type(x["allocated_bytes_upper_bound"]) is not int or
           x["allocated_bytes_upper_bound"]<0 or
           x["st_dev"]!=samples[0]["device"] for x in entries):
        raise ValueError("candidate allocated sizes or filesystem invalid")
    identities=[(x["st_dev"],x["st_ino"]) for x in entries]
    if len(set(identities))!=len(entries):
        raise ValueError("same recovery inode counted twice")
    available=[s["available_bytes"] for s in samples]
    low=min(available)
    required=max(floor_gib,planned_write_gib+reserve_gib)*GIB
    shortfall=max(0,required-low)
    potential=sum(x["allocated_bytes_upper_bound"] for x in entries)
    decline=max(0,available[0]-available[-1])
    duration=(samples[-1]["monotonic_ns"]-samples[0]["monotonic_ns"])/1e9
    return {
        "kind":"read-only root filesystem recovery candidate risk review",
        "sample_count":len(samples),
        "first_unprivileged_available_gib":round(available[0]/GIB,3),
        "last_unprivileged_available_gib":round(available[-1]/GIB,3),
        "lowest_observed_available_gib":round(low/GIB,3),
        "observed_net_decline_gib":round(decline/GIB,4),
        "elapsed_seconds":round(duration,3),
        "required_unprivileged_free_gib":round(required/GIB,3),
        "free_space_shortfall_gib":round(shortfall/GIB,3),
        "candidate_allocated_upper_bound_gib":round(potential/GIB,3),
        "candidate_upper_bound_could_cover_observed_shortfall":potential>=shortfall,
        "sampling_window_does_not_prove_capacity_stability":True,
        "actual_reclaimable_bytes_independently_verified":False,
        "candidate_consumers_and_open_handles_excluded":False,
        "archive_integrity_or_target_copy_verified":False,
        "safe_file_retirement_approved":False,
        "live_write_capacity_now_granted":False,
        "existing_research_job_guard_still_required":True,
        "files_moved_or_deleted":False,
        "candidate_count":len(entries),
        "candidates":entries,
    }


def inspect(root,paths,*,sample_count=3,interval=1.0,
            floor_gib=20,planned_write_gib=5,reserve_gib=8):
    if type(sample_count) is not int or not 2<=sample_count<=12:
        raise ValueError("sample_count must be 2..12")
    if type(interval) not in (int,float) or not math.isfinite(interval) or not 0<=interval<=5:
        raise ValueError("sample interval must be finite 0..5 seconds")
    root=_real_path(root)
    if not root.is_dir() or not paths:
        raise ValueError("an existing root and at least one candidate required")
    dev=os.stat(root).st_dev
    original=[candidate(p,dev) for p in paths]
    result=[]
    for n in range(sample_count):
        if n:
            time.sleep(interval)
        fs=os.statvfs(root)
        result.append({
            "monotonic_ns":time.monotonic_ns(),
            "available_bytes":fs.f_bavail*fs.f_frsize,
            "device":os.stat(root).st_dev,
        })
    final=[candidate(p,dev) for p in paths]
    if final!=original:
        raise ValueError("candidate modified or replaced during capacity observation")
    report=calculate(result,original,floor_gib=floor_gib,
                     planned_write_gib=planned_write_gib,reserve_gib=reserve_gib)
    # Cross-check the exact current unprivileged free policy used by jobs.
    current=preflight(root,planned_write_gib,reserve_gib,floor_gib)
    report["last_policy_preflight_available_gib"]=current["available_gib"]
    report["last_policy_large_write_allowed"]=current["large_write_allowed"]
    report["root_device"]=dev
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/"))
    p.add_argument("--candidate",type=Path,action="append",required=True)
    p.add_argument("--samples",type=int,default=3)
    p.add_argument("--interval",type=float,default=1)
    p.add_argument("--minimum-free-gib",type=float,default=20)
    p.add_argument("--planned-write-gib",type=float,default=5)
    p.add_argument("--reserve-gib",type=float,default=8)
    a=p.parse_args()
    result=inspect(a.root,a.candidate,sample_count=a.samples,interval=a.interval,
                   floor_gib=a.minimum_free_gib,
                   planned_write_gib=a.planned_write_gib,reserve_gib=a.reserve_gib)
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result["last_policy_large_write_allowed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
