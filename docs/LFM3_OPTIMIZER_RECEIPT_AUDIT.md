# LFM3 optimizer evidence: distinguish smoke steps from real benchmark runs

This module **does not run inference, finetune a model, or authorize
weight promotion**. It audits only the legacy unversioned JSON files
created by the two-step synthetic AdamW and Muon smoke experiments.

Run it from the auto-finetune repository root with existing local,
operator-owned receipts:

    python3 -m src.lfm3_optimizer_receipt_audit \
      --adamw /private/path/adamw/research_receipt.json \
      --muon /private/path/muon/research_receipt.json

Exit code 2 means BLOCKED, including valid legacy smoke receipts.
Unknown, private-text-bearing, malformed, nonfinite, oversized, linked
or promotion-claiming receipts are refused. No file names, raw messages,
full input JSON, checkpoint SHA values, or exceptions are printed.

The v1 output contains only the declared optimizer names, step counts,
trainable parameter count and categorical gate status. The inputs
include claimed adapter/base SHA-256 digests and a claimed disjoint
holdout loss, but these are NOT independently signed, witnessed,
source-verified, or evidence of real-world task performance.

Actual receipt check performed 2026-10-28 on x1-370:
AdamW: two synthetic steps; Muon: two synthetic steps; each reports
292,864 trainable LoRA parameters, and neither is production-ready.
Result: historical_synthetic_optimizer_smoke_only, BLOCKED.

## What is required to graduate

- Load actual pretrained LFM3 weights using the pinned loader and
  independent snapshot verification in draft PR #68.
- Run at least three independent *paired* training seeds under fixed
  token budgets and matching model/dataset/environment provenance.
- Report held-out task-level quality, relevant regressions, p95
  latency and peak GPU/system memory; score with a separately reviewed
  evaluator using disjoint and authorized test data.
- Submit the resulting numerical comparison to the separate paired
  holdout gate in draft PR #66. A passing synthetic manifest is
  **still not** authorization to deploy weights.
- Obtain authentic source/run/model/score custody receipts and
  rollback-reviewed promotion approval separately.

This legacy audit does not convert train-loss or a two-step heldout-loss
claim into a quality score, does not measure an optimizer winner,
and cannot attest the source checkpoint. It is intentionally
fail-closed on every legacy smoke test.
