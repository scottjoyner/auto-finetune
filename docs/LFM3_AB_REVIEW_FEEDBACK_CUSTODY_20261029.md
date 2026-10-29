# Read-only A/B review, registry and feedback custody (research draft)

Stacked on draft PR #71. All tests execute with synthetic/in-memory records
and temporary per-test storage; no production model registry, network,
inference services or training processes are accessed.

## Findings fixed

The legacy AutoPromoter.check_promotion method called
ModelRegistry.complete_ab_test before comparing feedback against minimum
quality thresholds. That method immediately activated its chosen winner,
which meant the subsequent "rollback" was an additional, non-atomic
activation—not a preventive review.

The new check_promotion method is a read-only evaluation. Once both
feedback arms have enough finite, correctly bounded samples and the
observation window matures, it returns action=review_required. It NEVER
invokes complete_ab_test or promote, modifies A/B state, or dispatches
automatic promotion alerts. A positive observation is *not* authority
to modify model weights or the active model.

ModelRegistry.record_ab_metric refuses NaN, infinities, booleans,
negative scores, score values greater than 1 and metric names that can
forge an internal *_count field. This restricts malformed totals
entering subsequent evaluation.

FeedbackCollector.record refuses any invalid score before persisting
input/output text. Feedback summary is idempotent across repeated
analysis, and flagged sessions are returned as unique string IDs.

AlertManager JSONL replay now collapses acknowledgement records by ID,
taking the latest appended state. UUID-based opaque identifiers avoid
same-millisecond alert/feedback collisions and same-second registry
version/A/B collisions.

## Verification and remaining holds

Focused tests validate invalid numerical data, one-sided samples,
minimum count, read-only evaluation, registry serialization/reload,
feedback repair-pair records, alerts across process restarts and mocked
webhook failures. Negative tests never transmit data or modify production.

The existing GitHub Actions 69-percent full-suite coverage threshold
remains unchanged. Do not lower it. Passing tests on a locally isolated
research branch is not enough to promote a checkpoint.

Critical remaining issues:
- Declared model family/identity is not independently attested.
- Registry.complete_ab_test still has a direct manual/CLI activation
  path for non-LFM3 models. That path needs explicit reviewed approval.
- Independently signed metrics, source trust and rollback are absent;
  no A/B improvement is proven.
- The feedback store retains original inference input/output up to
  2,000 characters. That data must not be copied into public reports.
- Appending JSONL updates without locks can still race across processes;
  a transactional source of truth is needed for production readiness.
- No deployed adapter, model weights, stop latches or services changed.
