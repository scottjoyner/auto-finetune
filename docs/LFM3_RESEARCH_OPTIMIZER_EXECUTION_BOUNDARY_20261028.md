# LFM3 optimizer feedback: execution-boundary and persistence hardening

**2026-10-28. Draft research only.** No production service, trained model, graph database, NAS, or external deployment modified.

## Why this boundary exists

The frozen original LFM3-1.7B was asked to review its own real final-FFN optimizer calibration. In two fixed prompt formats it voted **A (AdamW LR .001)**, even though a separately recomputed five-option three-seed experiment found **C (AdamW LR .004)** had substantially lower fictional development loss in every seed. The independent gate correctly vetoed the model's internally consistent but inaccurate advice. The later four-example parity-family audit favored C again, but is small synthetic evidence, not a broadly independent benchmark, and that holdout is **spent**.

This observation makes a strict separation necessary:
1. **Sensor:** real training losses, gradient norms, provenance, seed, optimizer state and task identity; untrusted until externally validated.
2. **Advisor:** original or adapter LFM3 may express a proposed optimizer/LR and reasons in a bounded, typed record. The advice is neither a metric nor an executable capability.
3. **Evaluator:** deterministic independent numeric checks and previously frozen dev/heldout protocol. Disagreement, invalid source or reused heldout produces a veto.
4. **Executor:** may run only a separately approved, bounded, scratch-isolated research trial with fixed model/source/snapshot and a new independent evaluation. Model text never invokes this directly.
5. **Promotion boundary:** a separately authenticated and audited operator decision, enforced at model serving. The existing registry's explicit boolean `operator_confirmed` is a *caller confirmation bit*, **not** authentication or signed authority; it MUST NOT be exposed as a capability to the advisor.

No single numeric gate or A/B metric can bootstrap a model into authorization to alter itself.

## Concrete hardening in auto-finetune PR #61

- `src/auto_promote.py`'s `check_promotion` **can no longer call `complete_ab_test`, `promote`, automatic rollback or any registry activation**, even when legacy `PromotionConfig(require_manual_approval=False)` is supplied. It emits `review_required` while preserving the running comparison. Manual promotion remains separate and requires further authenticated serving-level enforcement.
- A/B feedback keeps per-arm minimum sample requirements and time windows. Neither an overrepresented arm nor a low-quality advisor can bypass them.
- `src/model_registry.py` now uses collision-resistant opaque version and A/B IDs. It rejects overlapping active A/B tests on either model; never temporarily demotes a serving baseline to `testing`; and rejects malformed, nonfinite, negative or overflowing metric accumulators before recording data. Completion verifies both bounded score totals/counts and both registered versions before changing *any* A/B state. A valid explicitly confirmed completion saves one coherent state rather than promoting and later modifying it in multiple writes.
- The registry refuses corrupt persisted JSON, malformed arrays, duplicate version/test IDs and A/B records pointing at missing models. No malformed registry is silently replaced by an empty one. `AutoPromoter._load_state` likewise fails closed on corrupt/missing review fields instead of resetting the pending review and promotion history. The authoritative file remains untouched on a load failure.
- Regression tests are in `tests/test_registry_invariants.py`, `tests/test_model_registry_activation_gate.py`, `tests/test_auto_promote_review.py` and `tests/test_auto_promotion_nonexecution.py`. The first exact source passed 14 targeted tests on xwing, covering legacy opt-out, same-second ID collision, concurrent A/B enrollment, accumulator poisoning, non-mutation on invalid completion, and corrupted registry. The additional corrupt A/B review-state test is committed and requires exact-head CI verification.
- The existing minimum **69% aggregate Python test coverage** remains unchanged. This work does not make the PR production-ready. The last observed completed CI before these changes had approximately 58% coverage.

## Separate fixed online LR-switch research scaffold

`src/lfm3_ffn_fixed_lr_schedule.py` defines a **predeclared** synthetic-only training matrix: three fixed seeds; native AdamW and RMS-matched Muon; three genuine FFN LoRA gradient steps per arm; and four schedules per seed: each optimizer at fixed LR .004 vs first step .002 followed by two steps .004. Each optimizer state is retained across the predeclared LR transition. The frozen 1.7B backbone, fixed training examples, dev examples, model revision and rank-4 final-FFN target are pinned. No holdout is evaluated and no advisor text can alter the schedule. `tests/test_lfm3_ffn_fixed_lr_schedule.py` rejects widening the fixed matrix.

**Important status:** this schedule scaffold is committed but **has NOT been executed on a real pretrained LFM3 checkpoint in this pass**; it has no empirical performance receipt and must not be described as a demonstrated improvement, optimizer-control validation, or model-observed recursive loop. It cannot authorize live LR changes.

## SensorGraft journal lock is still unresolved

SensorGraft PR #7's fail-fast writer lock intermittently yields `Conflict("journal writer or reader lock is busy")` in the simple, sequential first-upsert integration test under four-thread test execution, despite distinct temporary sidecar pathnames and inodes. The failure was reproduced in **two of eleven** further stress executions using a test-only diagnostic that printed process ID, sidecar path, metadata, device number and inode. The production locking code and one-winner assertion were **not** weakened.

New failure-only test diagnostics also attempt to show `/proc/locks` entries and `/proc/self/fd` handles for the failing sidecar. This is instrumentation only, not a fix. Root cause and lock owner remain unverified; previously, instrumented syscall tracing had failed to reproduce the race. Source custody/witness issue #9 remains open.

## Required before controlled autonomous optimization

1. Real, repeatable optimizer-controller tests with an independently curated, held-back multi-family benchmark and per-seed task-level metrics; no repeated selection against spent parity or codebook holdouts.
2. Independent attested source provenance and consent, trustworthy source/time/version custody, and replay of sensor/observer/evaluator trace records in a robust SensorGraft journal.
3. Hard authorization at the serving/train-execution boundary with signed operator identity, quotas, anti-replay and human release review; remove any dependency on caller-provided `operator_confirmed=True`.
4. True process/crash/power-loss restart testing and multi-node ROCm repeatability; preserve frozen baseline and rollback before promotion.
5. Resolve the journal lock regression and restore required 69% coverage **without lowering thresholds or excluding production code**.

Keep both PRs draft. Do not deploy or enable unsupervised model-driven optimizer adjustment.
