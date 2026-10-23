# xwing checkpoint pruning custody hold — October 23, 2026

## Safety defect

The monthly crontab ran scripts/prune_checkpoints.py --apply on day 1,
at 03:30, which can delete optimizer, scheduler and RNG resume state.
The previous implementation iterated ALL checkpoint directories while
claiming in its header to preserve the most recent two checkpoints,
so it deleted resume files from even those newest two checkpoints.
No independent backup receipt was required. Historical xwing backup
logs include several read-only-mount failures and more recent 2.9GB
backups without proven coverage for every optimizer state.

## Corrective controls

- If launch-next.stop exists, any --apply invocation exits 78 BEFORE
  reading the checkpoint directory and BEFORE deletion. Dry-run is
  still available during hold.
- In all modes, preserve the two newest numeric checkpoint directories
  and all their resume data. Pruning targets only older checkpoints.
- Unknown checkpoint directory names and symlinked directories/labels
  are excluded, protecting rescue snapshots and outside destinations.
- The existing monthly --apply cron must also be commented out
  with a RECOVERY-HOLD note and backed up exactly. No checkpoint copy,
  pruning or deletion is authorized by this work.
- Verified per-file/source backup and recovery evidence are a separate
  prerequisite before re-enabling scheduled destructive cleanup.

## Synthetic checks

Seven isolated tests verify no deletion during hold, preservation of
two recent checkpoints in both standard and aggressive modes, dry-run,
explicit label keep, symlink and rescue-directory preservation.
The tests create and remove only temporary fixture data and never
touch real /media/scott/data checkpoint contents.

Follow-up: monitor stopfile, backups and physical R9700 recovery before
deciding to restore any automatic checkpoint cleanup.
