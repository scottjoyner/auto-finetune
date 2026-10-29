# Fail-closed promotion of LFM3 research candidates (draft)

This change is stacked on the admission/loader integration draft PR #70.
It is a safety hold, not an automatic adapter-release mechanism.

## Risk found in the preexisting registry and auto-promoter

The former AutoPromoter.register_and_test called ModelRegistry.promote
automatically when no current active model existed or when starting
the A/B test failed. ModelRegistry.complete_ab_test also automatically
activated the winning model. None of these operations established
official source identity, independent held-out quality, or authorized
promotion of LFM3 adapter weights.

## Changes

- ModelRegistry.promote refuses model versions whose declared
  base_model contains lfm3, without retiring the previous active
  model or saving new registry state.
- ModelRegistry.complete_ab_test refuses a winning LFM3 candidate
  before modifying the A/B experiment status, winner, end time, or
  registry state. Ordinary non-LFM3 completion remains unchanged.
- AutoPromoter.register_and_test retains LFM3 registrations as
  candidates for review, without attempting an A/B test or activation.
- For all model families, a missing active model or failed A/B start
  no longer silently promotes the new candidate; a separate explicit
  review and activation is required.
- Negative regression tests use Mock and in-memory ModelVersion
  instances exclusively. No production registry, checkpoint or service
  is read or altered.

## Not yet addressed

- A declared base-model name is not an authenticated source
  attestation. Mislabelled or aliased checkpoints remain a risk;
  a signed provenance chain and artifact inspection are required.
- Other non-LFM3 legacy registry paths still support automatic A/B
  winner promotion and need independent review before production use.
- This change does not provide a mechanism to approve LFM3 weights.
  The eventual release must use a separate reviewed approval receipt
  and rollback process; do not bypass this hold.
- Neither the paired-quality gate PR #66 nor the pinned checkpoint
  CPU-only smoke permits automatic promotion.
- The existing hosted 69-percent coverage gate remains mandatory.
  Do not lower it or merge this draft to avoid a failing CI result.

Tests are pure mocks and synthetic in-memory model versions.
