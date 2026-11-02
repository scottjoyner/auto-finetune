# LFM3 root filesystem recovery: candidate risk audit

**2026-11-02 | P0 storage issue #106 | draft PR #103 | READ ONLY**

## Newly verified measurements

On x1-370, the actual root filesystem `/dev/nvme1n1p6` (ext4) has fluctuated within approximately 13.5–14.2 GiB ordinary-user available space during this session. The latest four-sample read-only run of `src.root_recovery_candidate_audit` reported a low of **13.544 GiB**, a **6.456 GiB shortfall** from the established 20 GiB floor. The independent SSD `/media/scott/SSD_4TB` had about 415 GiB free, and the separately mounted CIFS `/nas` had about 35 TiB free. Space on a separate filesystem cannot directly free the root device.

Two potential archival recovery candidates exist as unmodified, single-link regular files on the **root** filesystem: `/var/tmp/scratch-snapshot.tar.gz` (~6.418 GiB) and `/var/tmp/scratch-data.tar.gz` (~2.008 GiB). Their combined **allocated-block upper bound is 8.426 GiB**, exceeding the measured 6.456 GiB shortfall. File modification dates were October 6. Their age, size and Linux file metadata do NOT establish disposability, archival validity, absence of open handles or future consumers. Earlier /tmp duplicate candidates from issue #106 no longer exist at their old paths; their fate is unknown. A prior `lsof` command emitted filesystem visibility warnings, so absence of positive file handles cannot be taken as proof that moving them is safe.

## Actual non-destructive SSD preservation completed

On 2026-11-02 the two root archive files were **COPIED, not moved**, to a new private owner-0700 directory on the separate research SSD: `/media/scott/SSD_4TB/lfm3_root_recovery_escrow_20261102`. The private destination has two owner-0600 single-link files with exact original byte sizes, 6,890,908,056 and 2,156,538,490. Each destination was hashed in full and matched the independently hashed SHA-256 baseline of its original. Both original files remained at their old `/var/tmp` paths with their original sizes/inodes; the SSD is a separate filesystem (`st_dev` 66305 versus root 66312). The copies did **not** free root space. No original has been deleted, renamed, truncated, overwritten or approved for retirement.

A subsequent `df` reading briefly suggested about 21 GiB unprivileged root headroom; a fresh seven-observation sample immediately afterward found a declining **19.231 → 19.137 GiB**, again below the 20 GiB floor. This variable external writer activity is not attributed to the preservation copies. No stable storage clearance or model-work authorization was established. The SSD copies protect against a loss of the two scratch archives, but their *content provenance, consuming services, future-path dependencies, archive extraction validity and safe original retirement* have NOT been independently approved. Checksum equality proves the copies match their originals, not that the archives are safe to remove.

## New reusable read-only admission

`src/root_recovery_candidate_audit.py` checks the target's actual stat device, each source file's actual `st_dev`, inode, size, allocated `st_blocks`, hardlink count, symlink ancestry and mtime. It rejects duplicate inode counting, cross-device supposed recovery and unsafe candidates. It samples ordinary-user `statvfs.f_bavail` with monotonic timing, calculates the lowest observed headroom and the shortfall against the identical 20 GiB floor and planned-write/reserve thresholds already used by `research_storage_preflight`. It separately re-runs the existing write preflight, and requires candidate metadata to match before and after observations.

The resulting report explicitly says **no file moved/deleted**, no live writer/consumer audit completed, no source integrity or archive copy confirmed, no actual reclaimable bytes verified, no retirement permission, no space reservation and **no model-run admission**. It does not override `research_job_guard.require_research_write_capacity`; the existing guard must run immediately before each substantial model-generation or benchmark operation. Even when candidate blocks exceed the shortfall, actual freed space can be lower or zero due to active handles, reserved files, concurrent writes and filesystem bookkeeping.

## Reproducible operator command

From the research directory on x1-370, without changing filesystem state:

    PY=/home/scott/lfm3-research/train-venv/bin/python
    "$PY" -m src.root_recovery_candidate_audit \
       --root / \
       --candidate /var/tmp/scratch-snapshot.tar.gz \
       --candidate /var/tmp/scratch-data.tar.gz \
       --samples 4 --interval 1 \
       --minimum-free-gib 20 --planned-write-gib 5 --reserve-gib 8

This read-only CLI exits with status 2 when the established write preflight still blocks research work. This is an expected safety result, not a program defect.

## Pending trusted-operator recovery requirements

Before any original retirement/deletion or further NAS/SSD migration, the owning administrators should verify actual service/job consumers, independently authorized archive location and permission/mount stability, the already completed full SHA-256 copy checks, actual consumer references, and any required archive content-validation checks. Only after external consumer authorization and verification should an owner decide whether to retire a redundant original. Re-audit real root available space and writer slope after any authorized change. Do not automatically kill signal-cli, change ext4 reserves, delete a cache, move either file, or resume model work.

The installed reviewer staging and 32-file private transfer receipt remain untouched. Independently provisioned real human reviewer accounts remain absent, and reviewer counts are still 0/1350 each. P0 storage issue #106 OPEN, human isolation #107 OPEN, draft PR #103 not release-ready.
