# Xwing Qwen dual recovery hold — October 24, 2026

## Incident and protective recovery

At approximately 14:34 EDT the primary xwing training stop-file
was present and training admission returned HOLD: stop_file, exit78.

At approximately 14:54 EDT the same file was unexpectedly absent.
Training admission still returned exit78, but only because memory
headroom was insufficient at that moment. **A resource shortage is
not a durable recovery hold**; if conditions improve, that admission
gate could otherwise authorize a training job before backups and
restore tests are complete.

At approximately 14:55 EDT the primary stop-file was recreated using
exclusive no-follow creation at:
  /media/scott/data/finetune-staging/launch-next.stop
File mode0600, non-symlinked. Admission returned HOLD: stop_file
exit78 again. No training process was found. All four commented
RECOVERY-HOLD boot/cleanup cron entries were preserved.
The responsible process for the disappearance was not identified.

## Independent second protection

To avoid any recurrence depending entirely on the same file, a
private independent recovery latch was created without overwrites:
  /home/scott/.fleet-custody/qwen-training-recovery.hold
The parent has mode0700; the latch is a regular, single-link,
owner-private mode0600 file. It is on a separate directory tree
from the fine-tuning staging drive.

src.training_admission.read_live enforces the following for xwing:

- If any primary stop-file directory entry exists, return HOLD:
  stop_file (including a malformed/symlinked entry).
- If the primary stop-file is absent, consult the independent latch.
- A valid owner-private latch returns HOLD: recovery_latch.
- Missing, unreadable, symlinked, world-readable, multiply linked,
  wrong-owner, or untrusted-parent latch states also return a HOLD.
- **There is intentionally no implicit release path** while this
  feature is active. Simply deleting either or both hold files
  cannot authorize a training process.
- Training admission on non-xwing fleet nodes is unchanged.

This is a defense-in-depth safety gate rather than a generic
resource-availability condition. A future training promotion requires
a separately reviewed explicit policy release and evidence of:
full independent 22-file backup, physical NFS durability,
actual sandboxed trainer-state restoration, compatible Python/ROCm
runtime and model/data provenance, and owner-bound GPU lease.

The change does not call an LLM or GPU, start a trainer, modify
checkpoint data, alter cron or inference services, or authorize
model deployment.

Offline fixture tests cover primary stop priority, missing primary
with intact secondary latch, missing both, symlink/corrupt latch,
untrusted directory permissions, non-xwing behavior, no-write guard
behavior and direct CLI exit78.

Related: auto-finetune issue #37. Keep BOTH holds in place.
