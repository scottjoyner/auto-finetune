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
