# Real pretrained LFM3 Dust prequential predictor benchmark — October 29, 2026

**Status: FAIL LEARNED-PREDICTOR ADMISSION, RESEARCH-ONLY.** No automatically selected optimization policy, trained checkpoint, promoted adapter, live model change, private trace ingestion or release.

## Motivation

The Dust/K2 forward-only training lineage (draft PRs #13, #22–24) supplied a perturbation estimator and direction predictor. LFM3 draft PRs #86–88 added a causal direction memory, prompt-activation-conditioned two-head predictor and an online-trained head-mixing gate, all backprop-free. PR #89 compared Dust, AdamW and factor-wise Muon on identical 2,048 LoRA-B scalar parameters, finding no three-step forced-choice task-accuracy gains.

Prior predictor experiments **tested only the directions recommended by each policy**. The observed short-horizon training gain therefore did not independently validate prediction skill: selected-only feedback can bias estimates, and repeated tiny selection sets cannot establish generalized improvement. This experiment explicitly separates **predict-before-measure** from **parameter update**.

## Pinned execution protocol

- Real official pinned pretrained BF16 LiquidAI/LFM3-1.7B on isolated x1-370 CPU, with the local Transformers compatibility patch and strict offline GPU-disabled environment. Pretrained base SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
- Per seed, **8 distinct procedural training prompts** (three seeds, 24 total). The originally disjoint selection and final-test partitions remain unopened. No private SensorGraft or customer data.
- At each episode, acquire a prompt-only four-scalar summary from LFM3's last-layer attention V-projection input. Generate four deterministic perturbation directions in a fixed eight-dimensional subspace of the same final two V-projection LoRA-B tensors used in PR #89 (2,048 scalars).
- Predict the gain of **all eight signed directions** (four plus and four minus) before measuring **any** of that episode's results, using five distinct forecasts: (1) always predict zero gain, (2) historical mean training gain, (3) direction-only historical attention, (4) prompt-activation-conditioned two-head attention, (5) online calibrated two-head attention.
- Execute all eight perturbed forwards regardless of policy choice. The mean-squared-error and top-ranked direction-gain comparisons therefore use the same measured candidates, avoiding selection-only measurement bias. Only after every signed direction has been forecast and probed are its scalar gains added to the head's memory for the **next** episode. Keep the head's 32-record rolling window.
- Restore the two LoRA-B matrices **by exact copy** in `finally` after every perturbed forward; assert initial/final SHA-256 equality. No backpropagation, gradient, torch optimizer, model adapter updates, model export or training-controller command.
- Exclude the cold-start episode from scoring, leaving **7 subsequent episodes × 8 directions = 56 prospective prediction cases per policy per seed**, or 168 per policy over three seeds. 64 measured candidate forwards per seed, 192 total, plus 24 prompt-context forwards and unperturbed clean-loss calculations. Do **not** equate these with a real-training efficiency comparison.
- The only adaptive weights updated are the **external two-coefficient attention mixer** (an online Hedge-style calibration rule) using prior training-only feedback. No transformer attention layer inside LFM3 was trained.

## Real measured outcomes

Relative mean squared error to predicting exactly zero gain for the same signed directions, where **less than 1 is better**:

| Policy | Seed 1 | Seed 2 | Seed 3 | Mean of seed-wise ratios |
| --- | ---: | ---: | ---: | ---: |
| Zero gain | 1.000 | 1.000 | 1.000 | 1.000 |
| Direction-only head | 1.440 | 1.054 | 1.824 | 1.440 |
| Activation-conditioned head | 1.415 | 1.053 | 1.501 | 1.323 |
| Online calibrated two-head gate | 1.456 | 1.053 | 1.644 | 1.385 |

Observed per-seed top-1 gain advantage versus the zero-forecast policy's stable first-candidate tie break:

| Policy | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: |
| Direction-only head | +0.004763 | -0.021496 | +0.000358 |
| Activation-conditioned head | +0.004783 | -0.012823 | +0.000358 |
| Calibrated head | +0.004783 | -0.010921 | +0.000358 |

These are descriptive gains **if** only the policy's top-ranked candidate had been probed within the four generated signed-direction pairs; they are not actual adapter updates or validation/test improvements. The zero-gain baseline is a deliberately cheap, fixed policy, not an optimizer. All three learned policies have **higher prospective MSE than zero on every seed**, and the seed-2 top-ranked candidate results make each learned policy's average top-1 gain advantage negative. There is no performance evidence to turn on learned optimizer direction selection.

## Automatic safety decision

`src/lfm3_dust_predictor_admission.py` checks the **strict structural report** for all three seeds, unique run IDs and matching non-test data inventories, then demands BOTH per-seed predictive MSE at least **5% better** than the zero-gain baseline AND strictly positive per-seed top-1 measured gain advantage. It additionally requires at least 32 truly independent source-level training episodes for even *nomination to a separate independent study*. The completed dataset has only 24 such training source prompts; directions and repeated same-example probes do **not** count as independent episodes.

All policies return **`hold_not_qualified`**. Even an eventual `eligible_for_separate_study_only` result would never grant permission to switch the deployed optimizer or change/promote model weights. No safety gate was bypassed or lowered.

## Evidence and limitations

Exact, immutable-by-Git, seed-specific scalar/hashes/direction-coefficient receipts are in `research/evidence/lfm3-dust-prequential-20261029/seed{1,2,3}-all-candidates-8episodes.json` (approx 40 KB per report). The independent CPU-only structural reviewer `src/lfm3_dust_prequential_review.py` checks exact example hashes, all positive/negative candidate coverage, normalization, 32-entry causal memory, episode-causal forecasts, bounded preprobe gate weights, expected calibration events and **recomputes every reported MSE, MAE and top-1 gain** rather than trusting report summaries.

A checked-in JSON receipt does not prove independent model-execution custody, actual prompt-only timing, model-freeze semantics or causal behavior under a malicious runner. The historical Dust/K2 shared-Unix-principal signing-key weakness remains unresolved. All examples are tiny procedural toy tasks and the benchmark uses a limited 8-dimensional perturbation subspace; these findings should **not** be generalized to larger real tasks or full fine-tuning without replication.

## Next experiments

- Before further model self-tuning, obtain **32+ independent training-source groups** and **new, genuine externally adjudicated task families**, with frozen validation and final-test splits outside controller feedback.
- Investigate why the calibrated head overpredicts rare perturbation gains; assess clipping, empirical shrinkage to zero and uncertainty-aware abstention in a *new preregistered* study, not by editing this frozen audit after seeing the outcomes.
- Compare the historical, zero, random, momentum and curvature-aware baselines on held-out, group-separated episodes, with total forward/backward, walltime, energy and cache costs recorded. If the head cannot pass a cheap forecast baseline, **use the cheaper method**.
- Keep all relevant branches draft and unmerged while full-repo coverage remains below the required 69% and independent reviewer approval, real task quality, retention, custody and Xwing admission controls remain outstanding.

## Reproduction

```sh
python -m pytest -o addopts= -q \
  tests/test_lfm3_dust_prequential_audit.py \
  tests/test_lfm3_dust_prequential_review.py \
  tests/test_lfm3_dust_predictor_admission.py

python -m src.lfm3_dust_predictor_admission \
  research/evidence/lfm3-dust-prequential-20261029/seed1-all-candidates-8episodes.json \
  research/evidence/lfm3-dust-prequential-20261029/seed2-all-candidates-8episodes.json \
  research/evidence/lfm3-dust-prequential-20261029/seed3-all-candidates-8episodes.json
```

The local real-model run requires the pinned checkpoint, explicit `--research-only --allow-forward-only-all-candidate-audit` flags, an exclusive new output path and all offline/CPU-disable environment variables required by the LFM3 probe. The verifier and CI run without model weights.
