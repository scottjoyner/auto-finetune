# xwing resumable checkpoint custody audit: October 23, 2026

## Read-only command

    cd ~/git/auto-finetune
    python3 -m scripts.resume_backup_audit

The command reads filesystem metadata and hashes component files ONLY
where both source and backup candidates exist. It never rsyncs, prunes,
releases data, loads models, writes to NAS, or alters GPU leases/power.
Machine-readable JSON is returned with exit 78 until an independent
restore test has been demonstrated; the audit is never launch authority.

It inventories numeric checkpoint-* directories; marks the two newest
per label for protection; and separates ABORTED/contaminated training
runs from production. The four candidate component file names are
optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json.
Symlinks and path traversal are rejected.

A backup generation is selected only when latest.txt is a valid timestamp
matching an existing directory. Components must be present in a precise,
versioned path below resume-checkpoints/<label>/<checkpoint>/ and match
source byte lengths AND SHA-256 digests. Missing backup candidates are
not needlessly hashed. Component hashes alone do not prove that a
restore can run: model and adapter weights, trainer configuration,
toolchain compatibility and dataset provenance may also be needed.
Therefore restore_test_passed, may_prune and may_release_source are
ALWAYS false even if every component hash agrees.

## Observed xwing findings

The source at /media/scott/data/finetune-staging/outputs/checkpoints has
30 checkpoint directories in two labels:
- qwen3-8b-sft-r1: 3 production candidate checkpoints
- qwen3-8b-sft-r1-ABORTED-contaminated: 27 quarantined checkpoints.
  The contaminated run must never be automatically promoted or used
  as clean scientific evidence.

All resume-related components occupy approximately 3.43 GiB.
The two newest production protected checkpoints are checkpoint-100
and checkpoint-150, totaling 350,134,973 bytes, about 0.326 GiB.

There are nine backup generations on the x1 SSD NFS mount, with the
latest pointer currently 20261019T081701Z. Existing backup generations
have final adapters but lack the scoped resume-checkpoints structure
and independent restore proof. Zero protected production checkpoints
have verified resume-component matches; verdict NO_VERIFIED_RESTORE,
exit78.

The weekly backup_ml_state.sh script explicitly excludes intermediate
checkpoint-* directories: its 2.9 GiB result proves neither resume
coverage nor a complete restore. Do not rely on latest.txt, transfer
size, or a mounted NAS path as proof of source custody.

## Next step

Before copying or testing a restore, independently verify destination
identity, isolation, space and applicable storage custody holds. Pin
exact source hashes and provenance; perform an explicitly staged
protected transfer and an independent restore trial. Never silently
turn on automatic transfer, training, source-release or checkpoint
pruning from the mere presence of backup directories.

The monthly destructive prune cron remains disabled under
RECOVERY-HOLD-20261023. All source checkpoints are preserved.
