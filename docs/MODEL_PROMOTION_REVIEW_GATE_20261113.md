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

## Atomic bucket publication and strict data-quality custody (2026-11-13)

This draft adds a distinct fail-closed acceptance boundary between synthetic
training records and any derived dataset artifacts. No live input, production
checkpoint, model process, credential, or node configuration was altered.

- Per-tool bucket export validates all source JSONL records before creating
  output; rejects malformed/empty records, dangerous tool names that would
  become filenames, missing source, empty datasets and invalid min bucket size.
- Bucket output is staged on the destination filesystem and published only
  after every file and manifest is written. Existing output/symlinks are not
  overwritten. On staged write failure the private staging directory is
  cleaned; no partial published manifest appears.
- Data-quality input no longer silently skips malformed JSONL: any invalid
  record holds the entire stage. Quality output cannot overwrite a prior
  artifact; a temporary file and no-clobber hard link protect complete
  publication. Missing/empty input, invalid threshold, no passing records,
  occupied destination or filesystem errors are HOLD.
- Real-time cleaning no longer treats failed quality validation as a skipped
  optional check. A failed gate or more than half the sessions failing quality
  blocks progression to formatting or training. Runtime errors record a
  generic status without leaking input content or credentials.
- The separate offline GitHub policy workflow exercises actual temporary
  files, malformed records, symlink/overwrite denial, stage cleanup and
  fake pipeline continuation. It never calls Neo4j, LFM3, physical GPUs,
  production keyfiles or deployed services.

The full 69% repository coverage requirement and all xwing/Joyner
recovery, GPU ownership, credential and operator promotion holds remain.
Passing local tests alone is not permission to deploy.

## Immutable QA dataset + provenance consumer contract — 2026-11-13

A read-only source trace found two separate production-invalid paths:

1. The previous real-time pipeline ran quality_gate during CLEANING, before
   the labeled formatter had written train.neo4j-traces.jsonl. A previous-cycle
   file could be mistakenly re-evaluated as the current output.
2. Even with a passing gate, src.cli train --source=neo4j-traces consumed the
   unfiltered train.neo4j-traces.jsonl, not the filtered QA output. Simply
   renaming filtered JSONL would also lose the row-aligned provenance sidecar
   required to evaluate benchmark holdouts.

This source-only draft now uses this strictly ordered chain:

  extraction -> clean -> format -> publish+verify immutable QA bundle
  -> verify QA source/receipt at should_train -> reverify at train
  -> trainer receives TRAIN_DATASET_DIR pointing ONLY to the vetted bundle

Bundle contract (schema auto-finetune.quality-bundle.v1):

- An identity derived from exact source and source-provenance bytes and the
  fixed 0.6 scoring policy determines the immutable directory name qb-<sha>.
- The directory contains train.neo4j-traces.jsonl, the exactly aligned
  train.neo4j-traces.provenance.jsonl, and receipt.json. Receipt contains input
  and output SHA-256 digests, policy version, source and accepted row counts.
- Missing, corrupt, mismatched, or untrusted provenance rejects the batch.
  Filtering preserves each accepted example's original session_id and assigns
  contiguous output row numbers.
- A private stage writes and fsyncs each file. An atomic no-clobber mkdir
  reserves the destination; receipt is published last. Partial/interrupted
  bundles are never trusted or silently overwritten. Exact-repeat input
  reuses a bundle only after complete byte-for-byte verification; changed
  source creates another identifier.
- Both the source and its provenance are rechecked at the training boundary.
  The trainer uses native train.<source>.jsonl naming inside the exact bundle
  directory so the standard holdout verifier sees the real sidecar.
- The direct real trainer now denies all statuses other than an independently
  CLEAN holdout result. Historically supported skip/unverified environment
  variables cannot authorize an actual train even if they change the logging
  of the diagnostic function. Dry-run is CPU-only and still rejects known
  contamination.
- The real-time child launcher strips legacy GPU and holdout bypass flags.
  A missing/invalid freshness check or a changed QA artifact is HOLD.
  The old hardcoded Qwen model override is removed; the real-time workflow
  denies a model config not containing LFM3 and does not fabricate a registry
  entry with a fixed checkpoint path or fake zero loss.
- Tests use only synthetic data under temporary directories, mocked
  subprocess results, and offline fake model identities. No model loads,
  user data read, remote Neo4j connection, GPU changes, credential accesses
  or production service changes were performed.

IMPORTANT REMAINING RELEASE REQUIREMENTS: a model family string containing
LFM3 is not enough to establish trusted pinned weights, exact trainer
checkout/runtime binding, secured benchmark custody or signed output
artifacts. A zero trainer subprocess exit is NOT checkpoint acceptance. Xwing
still requires its recovery/lease release decision; Joyner remains memory
constrained. The two draft PRs still need combined review and exact merged
commit CI. Existing production filtered outputs must not be relocated or
deleted merely to satisfy this source-only change.

## Independent trainer-boundary QA verification (Nov 13, 2026)

The orchestrator's prior verified-bundle-directory check is not sufficient
by itself: the source or symlink target might change before the separate
trainer opens the dataset. A successful orchestrator subprocess launch is
not independent trainer evidence.

- Direct src.train.main checks any data directory inside the reserved
  quality-bundles/ path independently, before importing a model library or
  loading a dataset. It requires the native train.neo4j-traces.jsonl filename,
  a matching approved source and label, and a complete integrity/quality
  receipt bound to the current original formatted source and its provenance.
- After dataset validation creates in-memory examples, the trainer checks
  the same source/bundle integrity a second time before choosing a backend.
  Broken hashes, altered JSONL/provenance, missing receipt, malformed path
  or invalid source binding return HOLD exit 78 before model loading.
- Quality-bundle publication and reuse refuse a symlinked quality-bundles
  parent directory to avoid writing outside the intended dataset namespace.
- Tests exercise only disposable paths and fake admitted decisions.
  Runtime credential, checkpoint, weight, lease and operator approval remain
  independent release gates. A local hash is not a signature or a statement
  about the authority that produced the source.
