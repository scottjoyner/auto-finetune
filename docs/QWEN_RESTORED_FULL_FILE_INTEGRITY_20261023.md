# Qwen full-file recovery candidate verifier

October 23, 2026. READ-ONLY. This verifies that a *future copy* of
the exact two protected full Qwen checkpoints matches the original
source digest manifest. It does not transfer any files, create a
backup, deserialize an optimizer, load a model or run a trainer.

## Trust boundaries

- The trusted source manifest comes from read-only live xwing hashing
  of 22 files totaling 547,737,523 bytes in checkpoint-100/150, as
  implemented in merged auto-finetune PR #52.
- One private source-hash manifest exists on x1-370 at
  /home/scott/.fleet-custody/qwen-manifests/full-source-manifest-20261023T2220EDT.json,
  mode 0600. The file's own SHA256 is
  366c2b9f327f8ec2d1359c428a20da8d7c76f95d0fdc91066560df3c24c2b8ed.
  This is a source evidence record, not a restore receipt.
- Before any real candidate copy, an independent operator must approve
  a non-production destination, root permissions, custody witness and
  capture of the exact 22 files; no transfer is authorized here.

## Offline usage against a later candidate copy

    python3 scripts/qwen_restored_file_integrity.py       --trusted-manifest /path/to/source-manifest.json       --candidate-root /path/to/disposable-restore-root

The candidate root contains the clean
qwen3-8b-sft-r1/checkpoint-100 and checkpoint-150 directories.
Both directories must contain **exactly** the 11 expected file
names from the manifest, with equal sizes and SHA256 digests.
Unexpected extra, missing, corrupt, linked or changed files
invalidate the candidate; path traversal and missing/malformed
manifests are denied. The script never reads the original source
checkpoint data, so it can be used in an independently isolated
restore staging area.

A byte-for-byte match means only FILE_HASH_MATCH_ONLY. The output
still says NO_VERIFIED_RESTORE, exit78, may_copy=false,
restore_test_passed=false, trainer_state_deserialized=false,
may_train=false, may_resume=false, may_prune=false,
may_release_source=false. The scanner has no write, model-loading
or transfer mode.

## Synthetic qualification and remaining blockers

Disposable fixtures create 22 tiny synthetic files, generate their
manifest, and copy to a separate temporary candidate directory.
Tests include equal-length checksum corruption, missing adapters,
unexpected files, symlinks, forged paths, untrusted manifest policy,
wrong steps, missing candidate root, read-only behavior and exit78.
NO live Qwen checkpoint was copied or restored to run these tests.

File integrity is necessary, not sufficient. The next independent
milestone remains an actual isolated **trainer restore test** with
base Qwen3-8B model identity, adapter, optimizer, scheduler, RNG,
trainer-state and dataset provenance, plus a deliberately gated
resume entrypoint. All current training and checkpoint-prune holds
remain in force; no source data can be released.

See auto-finetune issue 37.

The original private x1 witness was first saved under a symlinked
.local ancestor. This verifier correctly refused that ancestor as a
trust-boundary violation. An unchanged 3,667-byte manifest copy with
the same SHA-256 was therefore placed in the canonical owner-private
.fleet-custody directory, with mode 0600 on the file, mode 0700 on
its private parents, and no symlinked path components. The older
metadata witness remains unchanged. No source checkpoint was copied.

Live read-only check against this canonical manifest found
full_file_manifest_trusted=true, but the deliberately nonexistent
candidate restoration root still returned NO_VERIFIED_RESTORE,
candidate_file_count_matching=0, may_resume=false. The absence of a
real restored candidate remains the correct safety verdict.
