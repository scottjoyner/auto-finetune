# Model promotion requires recorded operator decision

**Draft, branch-only, research-only.** This proposal is stacked on
Auto-Finetune admission/credential/telemetry safety draft PR #108. It does NOT
modify running node services, active models, checkpoint files, training jobs,
production traffic, credentials, or deployment leases.

## Discovered safety defects

The previous start_ab_test routine silently changed the ACTIVE incumbent into
TESTING. The old complete_ab_test routine immediately called promote(winner)
and retired the loser. The auto-promotion controller then evaluated its score
and regression thresholds AFTER the winner had already been activated. An
unsuccessful A/B start, or no active baseline, could also trigger direct
promotion without any held-out review. That sequence conflicts with
explicit release and exact-head decision policy.

## Source-only changes

- Beginning an A/B study keeps the incumbent ACTIVE and marks only candidate
  versions TESTING.
- Completing a study records the winner in the isolated registry; it NEVER
  automatically activates or retires any version.
- Both arms must have independent finite and normalized quality scores.
  Missing/malformed metrics or self-comparison are HOLD.
- Auto-promoter registration can create a candidate and study, but NEVER
  defaults to activation when an incumbent or test is missing.
- Quality assessment requires balanced feedback and valid elapsed time.
  Eligible candidates return ready_for_review, not a production action.
  The check CLI returns exit 78 when review/HOLD remains.
- Direct manual registry promote still exists; this patch does NOT
  implement trusted approval receipts, signatures, or production credentials.
  A separate recorded operator authorization is required before invoking it.

## Tests and boundaries

Tests create a temporary on-disk JSON registry, synthetic versions,
fake feedback, deterministic scores and fake clocks. They verify that
registration, incomplete A/B studies, empty/imbalanced feedback, invalid
metrics, quality thresholds and evaluation cannot change incumbent identity.
Tests of explicit manual promotion use only disposable fixtures.

Do not merge, deploy or retarget model traffic without independent review,
exact-head hosted tests, the unchanged 69% coverage floor, approved dataset/
checkpoint custody, an operator decision record, and validated production
GPU ownership. Preserve xwing HOLD exit 78 and Joyner memory/GTT holds.

## Feedback, cost, sampling and export custody expansion (2026-11-13)

This is still source-only and stacked on draft PR #108. No source change here
is an operator authorization or a production deployment.

- Feedback admission rejects nonfinite or out-of-range scores, makes issue
  classification idempotent across repeated analysis, and returns distinct
  flagged session IDs rather than raw record objects. Scores and corrections
  in tests are synthetic only. Real feedback records remain private.
- Cost history now returns the most recent N observations, not the oldest N
  from a descending sort. Ledger writes use the existing atomic JSON writer
  rather than truncating the live file before rewriting.
- Bucket balancing uses a local deterministic generator (not process-global
  random.seed), rejects invalid weights/capacity, and refuses to claim success
  if selected session records are absent or the selected dataset is empty.
  Tests only create data under a disposable temporary directory.
- Model exports deny overwrite when output or GGUF merge-staging paths already
  exist. Synthetic fake model/PEFT/Torch/converter modules test the export
  branches without reading model weights, allocating GPUs or invoking a real
  conversion process. Existing output contents are preserved.

The dedicated offline Model A/B review workflow now exercises model
promotion, registry, feedback quality, cost ledger custody, bucket
balancing and model export safety with no attached credentials.

A/B winners still require independent review. Direct explicit registry
promotion still exists and needs an operator decision record in the
deployment process; this branch does not implement cryptographic approval.
The full suite remains subject to unchanged 69% coverage and every
existing production/credential/recovery/lease gate.
