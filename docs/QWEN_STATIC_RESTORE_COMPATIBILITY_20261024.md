# Qwen3-8B protected checkpoint static compatibility, October 24

**NO_VERIFIED_RESTORE. Training HOLD.** No GPU, model, pickle
deserialization, checkpoint copy or restoration involved.

The independent read-only script scripts/qwen_static_restore_inspect.py
checks the exact clean checkpoint-100 and checkpoint-150 directories.
It reads bounded JSON metadata and PyTorch ZIP CENTRAL DIRECTORY entries
only. It never reads the data.pkl member, invokes torch.load or pickle,
creates a new model, or starts a trainer. All status outputs preserve
NO_VERIFIED_RESTORE, exit78 and deny train/resume/prune/source release.

## Physical xwing readback

| Metadata | Checkpoint 100 | Checkpoint 150 |
| --- | --- | --- |
| global_step | 100 | 150 |
| max_steps | 2522 | 2522 |
| epochs | 2 | 2 |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| base model | local Qwen3-8B | local Qwen3-8B |
| optimizer ZIP entries | 1518 | 1518 |
| scheduler ZIP entries | 6 | 6 |
| RNG ZIP entries | 8 | 8 |
| training_args ZIP entries | 6 | 6 |

Both checkpoints passed this limited static structure check.
Controls reject missing, symlinked, hardlinked or unsafe JSON/ZIP,
global-step mismatch, unexpected base model, adapter rank/target
differences, tokenizer mismatch, ZIP member traversal, duplicate
members and oversized ZIP central-directory claims.

This is **not** a trainer-state restore, runtime/model compatibility
test, or an approval to copy or train. ZIP layout does NOT validate
pickle semantics, optimizer tensor shapes, serialized RNG
restorability or a valid training resume. This check does not inspect
the binary data.pkl content or prove its safety.

Repeat only read-only:

    python3 -m scripts.qwen_static_restore_inspect

Independent gates remain: explicit non-destructive production checkpoint
backup authorization, NFS no-replace and directory-fsync qualification,
complete 22-file copy verification, isolated/version-pinned deserialization
under a separately reviewed trust policy, actual optimizer/scheduler/RNG
restore, a guarded resume_from_checkpoint entrypoint and GPU leases.
The training stop-file and four cron recovery holds must remain in place.

Tracks scottjoyner/auto-finetune issue #37.
