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

## Neo4j credential source removal — 2026-11-13 follow-up

The real-time incremental Neo4j extractor previously embedded a literal
connection password in its source. The updated workflow **never** hardcodes
that credential: operators must supply NEO4J_TRACES_PASSWORD through an
approved runtime secret injection mechanism, without putting it in source
code, shell history, command arguments, issue comments, CI logs, or new
artifacts. No password value is included in this document.

If the variable is unset or empty, the extractor marks a
neo4j_traces_credential_unavailable HOLD, persists that generic reason,
does not launch a Neo4j child, and does not advance dataset progress.
The one-shot CLI now returns exit **78** on extraction HOLD, and nonzero for
failed cleaning/formatting. This prevents a scheduled retrainer from
misreporting a failed incremental extract as successful work.

When present, the secret remains in the inherited environment. The
orchestrator does not copy it into subprocess overrides or any argv.
Child stdout, stderr and Python exception messages are redacted **before**
logging or truncation; related tests use only synthetic fake strings.

**Important security boundary:** Removing the literal from current source
does **not** erase it from Git history, deployed worktrees, backups, or
existing log copies. The owning Neo4j administrator must independently
determine whether this credential is still active, rotate/revoke it if so,
and distribute a newly scoped secret via the approved manager. Rotation
and live service configuration require a separate authorized operation.
This draft does not establish that any connection succeeds; do not deploy
without owner-approved secret provisioning and admission review.

No real network/Neo4j connection, extraction, credential rotation, training
or production configuration change was made during this patch.
