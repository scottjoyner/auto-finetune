# LFM3 paired held-out research comparison

The CPU-only src/lfm3_paired_holdout.py evaluator compares at least three
paired seeds from independently generated baseline and candidate reports.
It is a strictly numeric, locally supplied, untrusted observation gate.

Usage, from the auto-finetune repository root:

    python3 -m src.lfm3_paired_holdout /path/to/private/paired-manifest.json
    python3 -m unittest discover -s tests -p 'test_lfm3_paired_holdout.py' -v

Input schema: sensorgraft.lfm3.paired-holdout.v1. Exactly these top-level
fields: schema, base_sha256, dataset_sha256, eval_sha256, benchmark_sha256,
baseline_optimizer, candidate_optimizer, baseline, candidate.

Each baseline/candidate array holds 3–16 independently trained run records:
seed, checkpoint_sha256, quality [0,1], p95_latency_ms (>0),
peak_memory_bytes (>0 integer). Records must have precisely matching seed
sets but different per-seed checkpoint digests.

The research-review thresholds are: paired mean quality gain >= 0.02,
no individual quality regression worse than 0.01, p95 latency ratio <= 1.25,
and peak memory ratio <= 1.15. These are provisional experiment policy,
not calibrated evidence of real-model improvement. A passing report is
candidate_for_independent_review, NOT a deployment approval.

The file reader refuses links, nonregular or oversized input, duplicate JSON
keys, nonfinite metrics and unexpected fields. The output never includes raw
prompts, completions or file paths. Inputs are self-reported and may be forged:
distinct hashes do not prove disjoint train/eval sources, provenance,
authentic hardware execution or cryptographic custody. A separate independent
evaluator must validate them, including test-set contamination, quality rubric,
latency environment, checkpoint digests and true model weights.

source_identity_verified, evaluation_independence_verified,
checkpoint_integrity_verified, and weight_promotion_permitted always
remain false. No model inference, training, network request, scheduler,
weight mutation or production service interaction is performed.
