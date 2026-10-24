# Qwen protected full-checkpoint custody plan

October 23, 2026, 10:22 PM EDT — BACKUP_PLANNING_HOLD. No transfer,
restore, training, model load, pruning, or checkpoint change performed.

The ten previously inspected dated backup generations in the two
configured SSD and NAS roots lack complete protected Qwen resume state.
The two clean production source checkpoint directories still exist
with all expected trainer components.

## Physical full-file inventory from xwing

Root: /media/scott/data/finetune-staging/outputs/checkpoints/qwen3-8b-sft-r1

| Protected checkpoint | Files | Full bytes | Four resume components |
| --- | ---: | ---: | ---: |
| checkpoint-100 | 11 | 273,861,396 | 175,060,121 |
| checkpoint-150 | 11 | 273,876,127 | 175,074,852 |
| **Total** | **22** | **547,737,523** | **350,134,973** |

Each checkpoint contains adapter_config.json, adapter_model.safetensors,
chat_template.jinja, README.md, optimizer.pt, scheduler.pt,
rng_state.pth, trainer_state.json, tokenizer.json,
tokenizer_config.json, and training_args.bin. The future backup must
cover the full set (about 522 MiB), not just four trainer-state files.

SHA-256 of all 22 files was streamed from xwing in read-only mode.
Sample adapter_model.safetensors digests:
- 100: 42577340442a089cfa0b5a114ffce45add2fd5b03ba62cae40cf2b2ebe738536
- 150: fc11841ab9fd8a3e8b26ac5bcd0f14c8547050fe5f85c60e76d2a876b628e69c

This is a point-in-time manifest, not an immutable snapshot or backup.
The verifier prints digest JSON but does not write output files.

## Operator read-only run

    python3 scripts/qwen_protected_backup_plan.py --hash-files

The report pins EXACT checkpoint steps 100 and 150 and rejects
silent substitution of older/newer steps; detects bad file types,
symlinks, hardlinks, missing critical files, size budgets and
same-device backup storage. Reported free SSD backup bytes:
589,551,370,240; conservative headroom estimate 2,169,216,870.
Capacity alone is not permission to transfer or evidence of custody.

Output always returns BACKUP_PLANNING_HOLD and exit 78, with
may_copy, may_resume, may_train, may_prune, may_release_source,
backup_verified and restore_test_passed all FALSE.
There is NO copy/execute mode.

## Independent gates

1. Approve an independent destination and custody owner for the two
   exact protected source checkpoints and retention plan.
2. Stage an atomic non-destructive transfer runner, with all 22 file
   hashes verified in a sealed target and a durable receipt.
3. Independently qualify model, tokenizer, adapter, dataset and code
   provenance, and source-state immutability.
4. Restore a disposable copy and verify actual optimizer, scheduler,
   RNG and trainer-state compatibility, not only equal file hashes.
5. Implement guarded resume_from_checkpoint behind stopfile, owner
   admission and GPU/resource lease, still requiring explicit action.
6. Preserve Joyner and xwing serving, all four boot/prune recovery
   holds, and quarantine of contaminated checkpoint data.

Tracking: scottjoyner/auto-finetune issue 37.
