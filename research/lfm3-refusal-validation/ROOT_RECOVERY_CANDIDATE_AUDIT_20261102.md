# LFM3 root filesystem recovery: candidate risk audit

**2026-11-02 | P0 storage issue #106 | draft PR #103 | READ ONLY**

## Newly verified measurements

On x1-370, the actual root filesystem `/dev/nvme1n1p6` (ext4) has fluctuated within approximately 13.5–14.2 GiB ordinary-user available space during this session. The latest four-sample read-only run of `src.root_recovery_candidate_audit` reported a low of **13.544 GiB**, a **6.456 GiB shortfall** from the established 20 GiB floor. The independent SSD `/media/scott/SSD_4TB` had about 415 GiB free, and the separately mounted CIFS `/nas` had about 35 TiB free. Space on a separate filesystem cannot directly free the root device.

Two potential archival recovery candidates exist as unmodified, single-link regular files on the **root** filesystem: `/var/tmp/scratch-snapshot.tar.gz` (~6.418 GiB) and `/var/tmp/scratch-data.tar.gz` (~2.008 GiB). Their combined **allocated-block upper bound is 8.426 GiB**, exceeding the measured 6.456 GiB shortfall. File modification dates were October 6. Their age, size and Linux file metadata do NOT establish disposability, archival validity, absence of open handles or future consumers. Earlier /tmp duplicate candidates from issue #106 no longer exist at their old paths; their fate is unknown. A prior `lsof` command emitted filesystem visibility warnings, so absence of positive file handles cannot be taken as proof that moving them is safe.

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

Before any move/deletion or NAS/SSD migration, the owning administrators should verify actual service/job consumers, complete file or archive checksums, authorized archive location and permission/mount stability, and conduct a checksum-verified copy **without changing the originals**. Only after external consumer authorization and verification should an owner decide whether to retire a redundant original. Re-audit real root available space and writer slope after any authorized change. Do not automatically kill signal-cli, change ext4 reserves, delete a cache, move either file, or resume model work.

The installed reviewer staging and 32-file private transfer receipt remain untouched. Independently provisioned real human reviewer accounts remain absent, and reviewer counts are still 0/1350 each. P0 storage issue #106 OPEN, human isolation #107 OPEN, draft PR #103 not release-ready.
