# x1-370 root capacity: verified Cargo incremental cache recovery

**Date:** October 31, 2026, approximately 12:41–12:48 EDT.
**Incident:** `scottjoyner/auto-finetune#106`.
**Scope:** operational recovery, not new LFM3 results or human grading.

## Capacity and mount verification

The affected root is `/dev/nvme1n1p6`, not the separately
mounted `/dev/nvme0n1p1` 4 TB SSD or `/nas`.
All eligible cache directories were verified with actual
`os.stat(path).st_dev` equal to root's device ID before reclaim.
The earlier 21 GiB Cargo cache cleanup under
`/home/scott/git/sensorgraft-db/target` occurred on the 4 TB
SSD and **is not counted as root recovery**.

Root unprivileged available space was **about 5.1 GiB** at
12:41; concurrent writers lowered one sample to **4.4 GiB**.
Two controlled cache reclamation passes removed these strictly
rebuildable Rust compiler incremental directories:

| Pass | Worktrees | Measured cache usage |
| --- | ---: | ---: |
| Git-ignored root Cargo incremental caches with no files newer than Oct 30 noon | 25 | 15.829 GiB |
| Additional root Cargo incremental caches last written before Oct 30 22:00, excluding core SensorGraft and active build | 12 | 5.305 GiB |
| **Total measured cache paths removed** | **37** | **21.134 GiB** |

Actual `df` recovery was checked after each pass rather than
inferred from `du`. Because other fleet jobs ran concurrently,
differences between snapshots cannot be assigned exclusively to
the cleanup. Following the second pass and ongoing concurrent
build activity, four samples taken three seconds apart showed
**26.071 GiB** of unprivileged root space available.

The research `src.research_storage_preflight` CLI returned
`large_write_allowed=true` under the unchanged **20 GiB
minimum, 5 GiB planned write, and 8 GiB reserve**; the separate
NAS mount remained available.

## Safeguards applied before removal

- Every target was precisely `target/debug/incremental` in a
  root-backed Rust worktree and marked Git-ignored. No `target/debug/deps`,
  compiled executables, benchmarks, models, reviewer data or source files
  were selected.
- No selected cache had files newer than its pass cutoff. Every
  candidate had to be a real directory without symlink path components.
- The inspected process command-lines and working directories showed no
  observed references to selected worktrees. The cache age and writer
  checks were repeated immediately before renaming each directory.
- Every cache was renamed inside its own directory and removed with
  `rm -rf --one-file-system` against only that renamed cache path.
- The active `sensorgraft-path-topk-20261030` Rust linker/compiler
  worktree and the core `sensorgraft` worktree were excluded.
  New 30–36 MB binaries in the former were observed during later checks.

## Remaining operational limitations

Cargo will regenerate incremental compiler data as required by
subsequent builds, incurring recompilation work. This action did not
stop other fleet processes or guarantee future free space.
A 26 GiB snapshot exceeds the guard by roughly 6 GiB but is not an
atomic space reservation, and active builds may consume that margin.
Issue #106 should remain open for monitoring and sustainable
build-target placement/retention policy, even though immediate root
exhaustion has been mitigated.

The frozen 1,350 XSTest private model responses, independent
human-review source CSVs, LFM3 checkpoint weights, and manuscript
evidence were **not edited**. Both human reviewer assignments
remain incomplete; no checkpoint was promoted or approved.
