# Qwen protected checkpoints — atomic, opt-in custody copy

October 23, 2026. Code and disposable tests only.
**Production transfer NOT YET AUTHORIZED or performed.**

The protected clean checkpoint-100 and checkpoint-150 source directories
contain 22 files totaling 547,737,523 bytes. Their source SHA256 manifest
was obtained read-only from xwing and retained as 3,667 bytes at
/home/scott/.fleet-custody/qwen-manifests/full-source-manifest-20261023T2220EDT.json,
SHA256 366c2b9f327f8ec2d1359c428a20da8d7c76f95d0fdc91066560df3c24c2b8ed,
mode 0600 under private, non-symlinked owner directories.

The current backup root on xwing is mounted over NFS from x1 at
/media/scott/SSD_4TB/agent-state-backups/ml-state-backups/xwing.
The backup mount reports ~550GiB free. Capacity is NOT transfer approval.

## No-write default and custody boundaries

scripts/qwen_atomic_custody_copy.py defaults to a read-only readiness
report and exit78. There is no implicit live copy. A future
operator-controlled execution additionally requires an explicit
--execute and exact approval phrase. The CLI refuses alterations to
the production source, backup root, stopfile, or training checkout.
The source manifest SHA256 must match the separately pinned digest.

Live training admission must report HOLD exit78 and the stopfile must
remain present before every file and checkpoint. Source and backup
must be distinct mounted filesystems, with conservative free headroom.
The destination is additionally pinned in /proc/self/mountinfo to the
NFS4 source 100.64.43.123:/media/scott/SSD_4TB at /media/scott/SSD_4TB;
a stale ordinary local folder after a mount failure is never sufficient.

All source files are checked by name, size and SHA256; the runner
rejects symlinks, hardlinks, altered source file metadata and
changed protected checkpoint selection. New data goes under a
random private .pending-qwen staging directory with directories
mode0700 and files mode0600 and exclusive no-follow file opens.

After the stage is copied and synced, all 22 resulting files are
independently rehashed against the source manifest, and only then
renamed to a new, unique UTC timestamp generation. No existing
generation or latest.txt backup pointer is overwritten or updated.
On failure, only this run's new private staging directory is removed.

The resulting generation contains:
- resume-checkpoints/qwen3-8b-sft-r1/checkpoint-100
- resume-checkpoints/qwen3-8b-sft-r1/checkpoint-150
- source-manifest.json
- custody-copy-receipt.json

The receipt never claims actual trainer restore: its status is
FULL_FILE_HASH_MATCH_ONLY, with trainer_restore_test_passed,
may_train, may_resume, may_prune, may_release_source all FALSE.
The CLI also returns exit78 after a verified copy, deliberately
preventing other automation from treating it as training clearance.

## Verification, future approval and remaining NO-GO

The synthetic test suite uses a private disposable /dev/shm backup
filesystem distinct from /tmp. It checks a full 22-file copy,
rollback, source immutability, file permissions, no latest pointer
mutation, stop-file revocation, bad admission, mismatches,
symlinks, wrong manifest and duplicate timestamp behavior.
No real checkpoint file or model is read in the tests.

An actual live transfer requires an explicit operator decision to
copy these two specific clean Qwen checkpoints to this custody root.
Merge/CI does not constitute that decision.

Independent requirements still open:
- Actual full backup and source-vs-copy SHA256 receipts
- Isolated optimizer/scheduler/RNG and trainer-state restore test
- Resume-from-checkpoint training entrypoint and model provenance
- Owner-scoped GPU resources and lease admission
- Preserve xwing inference, the training stopfile and all four
  recovery-hold boot/pruning entries
- Never release or prune checkpoint source from copy evidence alone

Tracked in scottjoyner/auto-finetune issue 37.

## October 24 concurrency and crash-consistency hardening

Previously a separate existence check preceded the normal rename,
creating a potential race in which a second writer could publish an
empty target directory before the copy stage was renamed. A normal
rename could overwrite that target. The runner now requires Linux
renameat2 with RENAME_NOREPLACE. An existing generation or symlink,
including one created in the final instant, is refused. Unsupported
kernel/filesystem rename behavior returns HOLD without fallback.

The runner fsyncs both new checkpoint directories, their parent
directories, and staging before publication, then syncs the final
backup-root directory after publication. A prepublication failure
removes only the private stage created by this execution. A failure
after publication leaves that candidate directory in place for
manual reconciliation, but explicitly refuses to certify the result.

All newly created dirs/files specify restrictive creation modes;
therefore the prior global process umask mutation was removed.
Tests exercise destination races with directories and symlinks,
unsupported atomic rename, before/after durability failures,
umask preservation, and concurrent publishers racing for one
generation. One publisher wins; the final synthetic file copy
can be verified by the separate independent 22-file checker.

IMPORTANT: the physical NFS4 destination has NOT been tested for
support of atomic no-replace or directory sync. Local synthetic
success is not evidence of NFS compatibility. Any actual production
transfer remains separately approval-gated. The runner must remain
HOLD rather than downgrade durability or overwrite protection.
