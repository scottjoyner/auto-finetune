# Model registry activation is closed pending an authorized release protocol

The legacy registry had direct CLI calls (promote and ab-complete) and
ModelRegistry.promote and complete_ab_test methods that could turn
self-reported feedback into active model routing without proof of
official checkpoint source, independent held-out evaluation, human
release approval or rollback custody. Checking the model family
string is not sufficient: names are untrusted and aliasable.

This draft **disables every activation path in this legacy registry**,
not just paths whose user-supplied model name contains LFM3.
ModelRegistry.promote returns false with no side effects.
ModelRegistry.complete_ab_test returns None with no side effects.
The matching CLI commands exit 78 (HOLD) before load() or model
registry file reads. No override flag or local environment bypass is
provided.

A/B initiation retains existing active models as active; candidate
versions may enter testing. Recording and reviewing bounded feedback
remain possible, but no feedback score can cause activation.

A future release mechanism must be implemented in a separately
reviewed change, using independently verified source/model/adapter
identities, metrics provenance, explicit approval signed by a trusted
release authority, one-time replay protection, persistent transaction
journaling and rollback. None of this is implemented here.

## Compatibility and limitations

This is an intentional breaking change: legacy direct manual
promotion and automatic A/B completion for all families now return a
hold. That can impact workflows expecting implicit activation, so
this must not be merged without independent review and release
operator approval. Existing deployments are NOT modified by this
branch. Old active entries can be observed; no new active records
can be produced through the public registry functions.

The registry is a JSON file, not a transactional system. Direct
modification of the file, monkeypatching Python state, or another
program writing an active entry remains outside the scope of this
guard. Process-level authorization and signed artifacts remain
independent blockers.

Testing: pure mocked values, temporary JSON files, CLI subprocess
entrypoints, and no GPU, production registry, network or weights.
The project-wide 69% coverage threshold is unchanged.

## Additional nonactivation transition controls

The original retire method could drop an existing active model without
an approved replacement. It now refuses retirement of active and
currently testing versions. The A/B start operation additionally
refuses identical model IDs, retired/testing inputs, two active arms,
and overlap with any running experiment. Two disjoint candidate
experiments may still proceed, with no model activation.

Unit tests cover these cases with frozen/temporary registries. These
guards are not a process-level transaction lock; independent writers
and cross-process registry updates still require an atomic journal
and external authorization before production use.
