# Xwing Qwen historical resume-backup coverage, 2026-10-23

**Verdict: NO_VERIFIED_RESTORE.** No training, pruning, source release,
checkpoint copying or resume operation is authorized by this document.

The prior backup audit checked the latest generation; the new
scripts/resume_backup_history.py scans multiple dated generations using
the existing SHA256 integrity checks. Each date stands alone: files
from different backup dates cannot combine to establish a complete copy.

## Verified physical evidence

The protected clean Qwen3-8B production run has two newest checkpoints:

- checkpoint-100: 175,060,121 bytes of four resume components.
- checkpoint-150: 175,074,852 bytes of four resume components.

Both source directories contain optimizer.pt, scheduler.pt,
rng_state.pth and trainer_state.json. The 27 checkpoint dirs under the
ABORTED-contaminated label are not eligible production restore sources.

All **nine** dated SSD-backed generations under
/media/scott/SSD_4TB/agent-state-backups/ml-state-backups/xwing,
oldest 20260824T211826Z and latest 20261019T081701Z, lack a
resume-checkpoints directory and each has **zero of two** protected
resume component sets verified.

The **one** known NAS-backup generation under /nas/ml-state-backups/xwing,
20260825T192118Z, also lacks a resume-checkpoints directory and
has zero of two protected resume component sets verified.

Thus all ten generations in these two *configured* backup roots
lack full resume-state custody. This does NOT prove copies do not
exist in other unknown NAS paths or media.

## Read-only repeat procedure

    python3 scripts/resume_backup_history.py --max-generations 12
    python3 scripts/resume_backup_history.py --backup-root /nas/ml-state-backups/xwing --max-generations 12

The code was run via authenticated SSH stdin on xwing, using its
existing base verifier SHA256
44c31afaf92a47cafcf439bd9b85418721977bd034de6dcb782566b244c4b2fb,
identical to the version in merged main. No script was installed on
xwing and no backup transfer or restoration was attempted.

All results deliberately return NO_VERIFIED_RESTORE, exit 78,
restore_test_passed=false, may_copy=false, may_resume=false,
may_train=false, may_prune=false and may_release_source=false.
Even complete component hash matches do NOT prove actual restoration.

Separate qwen_resume_preflight.py reports
training_code_has_no_resume_entrypoint, another independent blocker.

## Future gates, none approved by this audit

1. Design separately approved non-destructive backup of both clean
   protected checkpoints to an independent, custody-qualified target.
2. Verify full file and run provenance including base model, adapters,
   tokenizer, dataset and training-code versions.
3. Perform an isolated restore drill with cryptographic receipts and
   optimizer/scheduler/RNG/trainer-state compatibility checks.
4. Implement and test actual resume_from_checkpoint under stop-file,
   resource lease, post-boot and owner-bound execution guards.
5. Preserve existing training HOLD, quarantined contamination labels,
   inference services, and disabled destructive checkpoint pruning.

Evidence and gates tracked in scottjoyner/auto-finetune issue 37.
