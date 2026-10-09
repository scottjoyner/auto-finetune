# Auto-finetune CI dependency separation — 2026-10-09

**Draft, stacked on PR #35. No change to K2 estimator, source data, production runtime, GitHub runner credentials or coverage target.**

## Diagnosis and exact failure evidence

At [PR #35 exact head](https://github.com/scottjoyner/auto-finetune/pull/35), the focused new semantic + reviewer tests were green, but [general tests GitHub Actions](https://github.com/scottjoyner/auto-finetune/actions/runs/37996079646) failed during **pytest module collection**, before running any tests. GitHub log showed FOUR ImportErrors:

- `tests/test_bench_matrix.py`: `ModuleNotFoundError: No module named 'torch'`.
- `tests/test_compat_lfm2_rocm.py`: `ModuleNotFoundError: No module named 'torch'`.
- `tests/test_drivers_localchat.py`: `ModuleNotFoundError: No module named 'torch'`.
- `tests/test_quantize.py`: `ModuleNotFoundError: No module named 'transformers'`.

The hosted `.github/workflows/tests.yml` only installed `apsw pyyaml pytest pytest-cov`. The absence of heavyweight optional libraries is environmental, **not evidence that the four test modules failed their assertions**. Repeatedly collecting them as if all dependencies were installed wastes hosted CI and hides true coverage. The repository's `pyproject.toml` coverage gate is `--cov-fail-under=69` and remains unchanged; it must not be lowered merely to force a green run.

## Implemented split

**Hosted `.github/workflows/tests.yml`** is explicitly named `core CPU tests (model integrations separate)`. It installs the same lightweight prerequisites and runs the standard test suite with explicit `--ignore` for *only* the four model-dependent modules above. The workflow writes its exact exclusions to the GitHub job summary, so a green core job **cannot be presented as passing all tests**. The original 69% `src/` coverage ratchet remains active; if omitting model test coverage makes the threshold fail, that is an honest finding and requires additional independent CPU tests—not hiding/weakening the threshold.

**Manual `.github/workflows/model-integrations-local.yml`** describes an opt-in, **trusted-main-only** `workflow_dispatch` job pinned to the `[self-hosted, linux, x64, research-model-tests]` runner label. It uses locally installed Torch/Transformers and explicit `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE` and runs only those previously uncollected tests. There is **no automatic self-hosted PR execution**, no hosted model checkpoint downloads, no fallback to a generic shared fleet node. It is not activated until a physically dedicated self-hosted runner is configured with trusted credentials, immutable package dependencies and safe local temp storage; **no runner or account was installed by this PR**.

The optional workflow is not an independent signer or production deployment proof.

## Actual local Xwing validation

Using the preexisting Xwing private ROCm research Python environment (Torch `2.12.0+rocm7.14.0`, Transformers `5.18.0`), the four model-dependent modules could be collected without changes: **83 tests collected**.

Then ran locally under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`:

```bash
python -m pytest -o addopts='' -q \
  tests/test_bench_matrix.py \
  tests/test_compat_lfm2_rocm.py \
  tests/test_drivers_localchat.py \
  tests/test_quantize.py
```

**Result: 82 PASSED, 1 SKIPPED, 0 FAILED** in approximately 3.40 seconds. These are REAL assertions, not synthetic claims about the original hosted collection failure. The skipped case should remain reported as skipped; skip reason and reason for any future missing test must not be concealed.

Validation ran in the isolated Git worktree of PR #35, not in a production deployment or NAS recovery filesystem. No K2 optimizer training, model weight updates, credential provisioning or new production service were initiated.

## Gate distinctions

- Hosted **core** tests: standard core suite, explicit 4-module exclusions, original 69% coverage threshold. Not a full integration suite.
- Xwing local targeted model tests: **82 pass, 1 skip**, real Torch/Transformers environment; not a GitHub runner or release gate.
- New self-hosted model-test workflow: manual-only and guarded to trusted main; **NOT_RUN**, runner absent/unproven until separately installed.
- Signer key isolation: prior legacy x1 signing key readable under the producer-shared Unix `scott` account, separate UID service still NOT_VERIFIED.
- Source dataset rights, independently human-adjudicated semantic challenge, real K2 feature cohort minimum, classifier-vs-momentum superiority, optimizer/production authorization remain **HOLD/DENY**.

**Disposition: exact hosted missing-dependency cause CONFIRMED; 82/83 model assertions PASS locally; hosted core CI scope separated without relaxing the coverage ratchet; self-hosted local runner registration and production use NOT AUTHORIZED.**
