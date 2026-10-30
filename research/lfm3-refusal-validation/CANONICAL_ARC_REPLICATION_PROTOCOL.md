# Canonical EleutherAI ARC-Challenge full evaluation

Research protocol for the unchanged builtin arc_challenge task in
EleutherAI lm-eval 0.4.13, zero-shot, exactly the original frozen
A/B/C LFM3-1.7B checkpoint files. This is separate from the already
finished full local-parquet adapter study.

## Canonical data now available (2026-10-30)

The exact pinned Hugging Face data source allenai/ai2_arc,
revision 210d026faf9955653af8916fad021475a3f00453,
config ARC-Challenge, loads successfully on x1-370 with 1,172
test questions. Read-only comparison against the local parquet
confirmed all 1,172 rows identical and in the same order,
including IDs, question text, choices and answer keys.
The SHA256 of canonical JSON serialization (sorted keys,
no whitespace, UTF-8) is
de86c368d3a59701a5114595cefc1869edac18bd558710f930bf6b6a7a753224.
The original parquet SHA256 is
62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9.

The isolated cache is on the 4 TB SSD at
/media/scott/SSD_4TB/hf-cache/huggingface,
not on the nearly-full root filesystem.

## Private canonical scoring contract

The built-in canonical arc_challenge task must execute all
1,172 cases with --num_fewshot 0 --batch_size 8 --device cpu,
pinned BF16 model paths, and no --limit. Use --log_samples
so row identity, per-case correctness and aggregate means
and standard errors can be checked. Do not publish raw
sample JSONL or private result JSON.

A full three-arm provenance verifier is implemented in
src.arc_canonical_full_report. It requires exact official
task and parent YAML hashes, frozen checkpoint SHA256s,
complete logs, identical question content and order, and
reconciled mean/stderr for both acc and acc_norm. It
compares outcomes against the previous full local-parquet
runs and calculates matched exact paired tests with
six-contrast Holm correction.

The public aggregate report may ONLY be published after
all real runs finish and strict verification succeeds.
No automatic grading, model-weight edits, dataset substitution,
or benchmark completion claims from an incomplete run.

Canonical capability scores cannot establish behavioral
refusal, suitable harmful-action restrictions, or safe
release. Two independent human XSTest rating assignments
remain the separate scientific and release bottleneck.


## Execution and strict audit completed (2026-10-30)

All three unmodified builtin arc_challenge full-task runs have now
COMPLETED, using the exact original A/B/C frozen checkpoints.
The private cases were reconciled against the pinned parquet:
3,516 / 3,516 matched the source questions, and canonical
acc/acc_norm binary outcomes agreed with the earlier full
local-adapter evaluations with ZERO mismatches.

See CANONICAL_ARC_RESULTS_20261030.md and the public aggregate
evidence/20261030_arc_canonical_full1172_cpu_aggregate.json.
External independent host reproduction and behavioral human
review are still outstanding, and model release is blocked.
