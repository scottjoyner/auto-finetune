# Qwen3-8B resume dependency preflight — October 23, 2026

The separate resume-backup audit found ZERO matching protected
production checkpoint component backups. This preflight checks the
current source checkpoint, indexed base shards, trainer implementation
and clean dataset split without running any workload.

Read-only commands on xwing:

    cd ~/git/auto-finetune
    python3 -m scripts.qwen_resume_preflight
    python3 -m scripts.qwen_resume_preflight --hash-partition-source

No torch model load, binary deserialization of training_args.bin,
optimizer restore, GPU probing, NAS write, or training starts.
Optional SHA-256 streams only the local source split JSONL and checks
its bytes against the stored source_sha256 manifest.

All results remain NO_GO exit78, and may_resume, may_train, may_copy,
may_prune, may_release_source, and restore_test_passed are ALWAYS false.
Passing local metadata checks is not a verified checkpoint restore.

## Physical xwing evidence

The clean Qwen3-8B checkpoint-150 contains an adapter, tokenizer,
optimizer, scheduler, RNG, training args and trainer_state reporting
global_step=150 / max_steps=2522. The adapter config names local base
models/Qwen3-8B, and all 5 indexed safetensors model shards exist.
File presence does not validate complete base weight hashes.

The mixed-seed42 partition records clean zero held-out overlap,
20,163 train examples and 2,240 eval examples. Its absolute source,
train and eval paths all exist. SHA-256 of the 417,373,429-byte
local source file matches the partition manifest hash.

Both source training backends in src/train.py currently call
trainer.train() without resume_from_checkpoint. The ACTIVE xwing
src/train.py contains additional local edits which were not overwritten.
No current code-level resume entrypoint has been demonstrated.

The older qwen3-8b-sft-r1-ABORTED-contaminated run remains quarantined
from clean production restoration and benchmarks.

## Still required

Preserve checkpoint files, recovery stop-file, three boot-time holds
and disabled monthly destructive pruning. Verify separately-custodied
complete checkpoint backups with cryptographic receipts. Implement
an explicitly gated and tested trainer resume pathway; prove a
disposable restore trial with exact model/config/dataset provenance.
R9700 OCuLink hardware identity, GPU lease and resource headroom are
still independent production gates. No automatic transfer or launch
is authorized by this preflight.
