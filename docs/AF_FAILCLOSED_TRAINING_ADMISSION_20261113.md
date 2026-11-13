# Auto-Finetune fail-closed admission reconciliation — November 13, 2026

## Decision

**Draft/source-only. No deployment, training restart, model loading, credential
changes, lease release, recovery-marker writes, scheduled jobs, or production
rollout.** Pinned base: `bce80eafee42105c44b7ea5390f6b35fa2fca56d`.

The released `main` source contains `AF_SKIP_UNMANAGED_CHECK=1` admission
shortcuts in the training guard, direct trainer and CLI, and launchers that set
the same variable. `tests/conftest.py` also globally injects it.
The xwing local checkout at `908fcd6` has a narrower guard, but is
not proof that published/main callers cannot bypass admission.

## Scope of this source-only patch

- Remove `AF_SKIP_UNMANAGED_CHECK` escape paths from
  `src.training_admission.read_live`, `src.train.main`, and CLI unmanaged
  training arbitration. The direct trainer still skips GPU admission for a
  CPU-only `--dry-run`; real training always checks it first.
- Defer heavyweight Transformers import side effects until **after** admission,
  allowing model-free policy tests to run without GPU dependencies.
- Refuse malformed/unavailable CLI admission and unknown unmanaged-trainer
  census rather than allowing a GPU/dataset lease.
- Never inject or inherit `AF_SKIP_UNMANAGED_CHECK` or
  `TRAIN_ALLOW_UNVERIFIED` from the scheduled, realtime, or bucket launchers;
  resolve the bucket runner's cwd from its actual checkout.
- Treat nonzero subprocess results in scheduled retraining as failure (and
  keep a failed cycle out of success statistics); a failed GPU inventory
  process check also denies scheduling.
- Remove the test-suite-wide admission override. Synthetic dispatch tests
  already mock the exact policy/lease functions they exercise.

## Evidence and limitations

The lightweight local admission/scheduler tests include deliberately hostile
override-variable and subprocess-error cases, with real model/server calls
prohibited. The existing source tests cover the current shell/lease guard.
The normal full hosted test suite retains its **69% coverage floor**. The
lightweight local test environment lacks `apsw`, so full-suite claims require
hosted exact-head CI with its declared package dependencies.

Residual HOLD: the direct `TRAIN_ALLOW_UNVERIFIED` dataset provenance waiver
still exists in the core trainer, although these launchers no longer supply it.
Any dataset integrity policy revision needs a separate owner/security review.
This PR does not clear the xwing immutable/absent recovery latch, occupied GPU,
protected checkpoint restore requirements, or R9700 RAM/swap/GTT thresholds.

## Acceptance for merge or deployment

1. Complete exact-head hosted xwing admission and general tests with no
   coverage reduction or disabled test collection.
2. Review all entrypoints, the residual dataset-provenance flag and managed
   scheduler/cron wrappers; ensure no other bypass routes are deployable.
3. Preserve xwing `HOLD` exit 78 and production credential boundaries.
4. Obtain separate operator/lease/custody decision before a real GPU training
   run or promotion.

Rollback: no running services altered. Close the draft and discard its
isolated worktree if the safety review rejects the patch.
