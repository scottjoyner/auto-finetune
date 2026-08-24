#!/usr/bin/env python3
"""Prune dead weight from finished training runs.

Policy:
  * ACTIVE run (checkpoint dir touched recently or named on the command
    line via --keep) is left untouched.
  * For every other label dir: keep the FINAL adapter files + last 2
    checkpoints intact; delete optimizer.pt / scheduler.pt / rng_state.pth
    from older intermediate checkpoints (resume-impossible but eval/merge
    of those snapshots never happens anyway).
  * --aggressive also deletes older intermediate checkpoint dirs entirely
    (keeps last 2).

Dry-run by default; pass --apply to delete.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

BASE = "/media/scott/data/finetune-staging/outputs/checkpoints"
RESUME_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth")
KEEP_CHECKPOINTS = 2
ACTIVE_HOURS = 6.0  # dirs modified within this window are considered active


def is_active(label_dir: str, keep: list[str]) -> bool:
    if any(os.path.basename(label_dir) == k for k in keep):
        return True
    try:
        age_h = (time.time() - os.path.getmtime(label_dir)) / 3600
    except OSError:
        return True
    return age_h < ACTIVE_HOURS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep", action="append", default=[],
                    help="label dir names to never touch (repeatable)")
    ap.add_argument("--aggressive", action="store_true",
                    help="also delete older intermediate checkpoint dirs")
    args = ap.parse_args()

    freed = 0
    for label in sorted(os.listdir(BASE)):
        ldir = os.path.join(BASE, label)
        if not os.path.isdir(ldir) or is_active(ldir, args.keep):
            print(f"[prune] skip active: {label}")
            continue
        ckpts = sorted(
            (d for d in os.listdir(ldir) if d.startswith("checkpoint-")),
            key=lambda c: int(c.split("-")[-1]) if c.split("-")[-1].isdigit()
            else 0,
        )
        old_ckpts = ckpts[:-KEEP_CHECKPOINTS] if len(ckpts) > KEEP_CHECKPOINTS \
            else []
        for c in ckpts:
            cdir = os.path.join(ldir, c)
            if c in old_ckpts and args.aggressive:
                sz = sum(f.stat().st_size for f in os.scandir(cdir)
                         if f.is_file())
                print(f"[prune] {'DELETE' if args.apply else 'WOULD DELETE'} "
                      f"dir {label}/{c} ({sz/1e6:.0f} MB)")
                if args.apply:
                    shutil.rmtree(cdir, ignore_errors=True)
                freed += sz
                continue
            for fname in RESUME_FILES:
                fpath = os.path.join(cdir, fname)
                if os.path.isfile(fpath):
                    sz = os.path.getsize(fpath)
                    verb = "DELETE" if args.apply else "WOULD DELETE"
                    print(f"[prune] {verb} {label}/{c}/{fname} "
                          f"({sz/1e6:.0f} MB)")
                    if args.apply:
                        os.remove(fpath)
                    freed += sz
    print(f"[prune] total {'freed' if args.apply else 'recoverable'}: "
          f"{freed/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
