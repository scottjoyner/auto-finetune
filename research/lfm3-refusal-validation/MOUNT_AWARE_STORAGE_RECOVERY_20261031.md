# Mount-aware, read-only LFM3 storage recovery inventory

**2026-10-31, x1-370.** Scope: incident #106. This inventory does
not remove, modify, migrate or reserve any files and does not invoke a
model, a compiler, or a training job.

## Incident-specific discovery

Root `/dev/nvme1n1p6` is only one of several mounted filesystems
under seemingly related paths. The directory
`/home/scott/git/sensorgraft-db/target` looked like a large root
worktree but is actually backed by the separate 4 TB SSD
`/dev/nvme0n1p1`. Its stale, Git-ignored 21 GiB
`target/debug/incremental` Rust compiler cache was cleaned
after an inactivity check, but **that did not free root space**.
This was a mount-attribution error; the compiler cache will need
regeneration if that worktree is rebuilt. No source files, model
weights, benchmark reports or user data were intentionally removed.

A recheck found **5.237 GiB** of ordinary-user available root space
against a **20 GiB** heavy-evaluation floor: **14.763 GiB short**.
The `/nas` mount was still separate with roughly 36 TB available.

## Mount-verified candidate mapping

| Candidate location | Backing filesystem | Can affect root? |
| --- | --- | --- |
| `/tmp` | `/dev/nvme1n1p6` | yes, if safely reclaimable |
| `/home/scott/.cache` | `/dev/nvme1n1p6` | yes, if safely reclaimable |
| `/home/scott/git/sensorgraft-db/target` | `/dev/nvme0n1p1` | **no** |
| `/var/log` | `/dev/nvme0n1p1` | **no** |

This is **not** a deletion list. Other active fleet sessions, scratch
files, logs, compiled executables and checkpoints must be preserved
unless custody and reconstruction are proven. Large `du` size alone
does not mean those bytes can or should be freed.

## Read-only reproduction

From `research/lfm3-refusal-validation`:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
$PY -m src.mount_aware_recovery_audit \
  --target / \
  --candidate /tmp \
  --candidate /home/scott/.cache \
  --candidate /home/scott/git/sensorgraft-db/target \
  --candidate /var/log
```

The `--measure-usage` flag optionally calls `du -s -B1 -x` to
observe usage on each candidate's actual backing device. The tool
always reports `automatic_cleanup_authorized=false`, never reports
an amount as safely reclaimable, and **never deletes anything**.
`os.stat(path).st_dev` determines whether the candidate can affect
the target's `statvfs.f_bavail` value, not pathname matching.

## Remaining operational blockers

The minimum 20 GiB headroom is still unmet; no heavy LFM3 runs
should be started. A privileged administrator may separately inspect
actual ext4 reserved-block policy, but nothing here changes it.
Issue #106 remains open. This module does not prove a candidate is
unused, recoverable, privately owned, inactive or safe to dispose of.

The XSTest full source journal remains frozen at 1,350 outputs.
Both independent human reviewers still have 0/1,350 graded responses,
and draft PR #103 has no authorized model-safety or release decision.
