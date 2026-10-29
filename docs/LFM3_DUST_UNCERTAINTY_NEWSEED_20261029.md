# LFM3 predictive Dust uncertainty gate — out-of-seed forward-only research, Oct 29, 2026

**Decision: HOLD.** A backprop-free analytic online shrinkage controller can observe actual LFM3 prompt activations and adapt its forecast amplitude to measured train-only forward outcomes. Its conservative uncertainty veto withheld every hypothetical recommendation in a new seed cohort. The method **did not beat a zero-gain predictor** and is **not approved to switch optimizers or update production model weights**. All work is isolated, draft research.

## Relationship to earlier predictive fine-tuning

The older Dust/K2 work (draft PRs #13 and #22–24) defined forward-only perturbation estimation and direction prediction, with negative historical confirmations. The LFM3 work in draft PRs #86–88 implemented causal pre-probe direction attention, prompt-only LFM3 activation context, and an analytically updated two-head attention mixer. PR #89 compared AdamW, experimental factor-wise Muon and Dust on **the same 2,048 final V-projector LoRA-B adapter parameters**, finding no improved three-step forced-choice task accuracy; gradient computation budgets were unequal. PR #90 performed the first three-seed all-candidate *prequential* forecast audit: 192 real perturbed forwards, with every learned head worse than simply predicting zero gain.

This branch extends that line with a new **cheap scalar shrinkage and abstention controller**, plus **new seeds 4–6**. It does not yet insert or train a neural transformer attention head *inside* LFM3. The adaptive controller is an external predictive module whose small online coefficients can change from historical forward-only evidence. No backpropagation is needed for its coefficients.

## Frozen algorithm, defined before running seeds 4–6

Source `src/lfm3_dust_uncertainty_gate.py` learns one real number `alpha` for each existing prediction head, constrained to `[0,1]`. Before episode `e`, using only earlier training-example directions and gains:

`alpha = clip(sum(raw_pred_i * measured_gain_i) / (0.01 + sum(raw_pred_i**2)), 0, 1)`

This is ridge-regularized analytic shrinkage, not gradient descent. Each new signed-direction prediction becomes `alpha * raw_pred`. A conservative **heuristic** estimate `lower_bound = shrunk_pred - 2 * RMS_training_residual` is computed from prior episodes only. It is **not a statistically valid confidence interval** or proof of high-probability improvement: sparse correlated probes and optimistic in-sample residuals can miscalibrate it.

The gate can suggest a probe *only if* at least four complete prior training-source episodes exist, its historical mean-squared error is 5% better than always predicting zero, and the lower bound is strictly positive. It never proposes or enacts an optimizer switch or LoRA update, and any future experiment must separately confirm the gain by a fresh forward pass within the model's trust radius. Source labels from selection, final-test, private or future episodes are rejected.

## New actual pinned-model experiment, distinct seed cohort

Pre-registered code was committed before collecting new LFM3 data on the x1-370 fleet node. Official pinned pretrained BF16 LiquidAI/LFM3-1.7B, strict offline CPU-only execution, final two attention `v_proj` LoRA-B matrices (2,048 scalars). Seeds **4, 5, 6** each generated 12 procedural training examples (a total of **36 distinct family/operand keys**; all examples are from the same three synthetic task families). Distinct seeds are not the same as independent real-world evaluation.

Per seed, evaluate four fresh candidate perturbation directions in both signs for every training prompt, always measuring all eight directions irrespective of predictor preference. This yields:

- **288 actual perturbed model forwards** (3 × 12 × 8), 36 additional prompt-only activation-context forwards and clean comparison passes.
- Exactly **36 training-source prompt episodes**; individual perturbation directions on one prompt are correlated and must not be called independent training examples.
- One cold-start episode per seed excluded. **33 new-episode forecasts per head**, each with eight signed directions, so **264 prospective forecast cases per head**.
- No model gradients/backpropagation, no cumulative adapter changes, no optimizer invocation, no checkpoint exports, no selection or final-test scoring, no private user traces. Every ± perturbation is restored by exact-copy `finally` and the initial/final LoRA-B hash matches.

Original complete source receipts are preserved in `research/evidence/lfm3-dust-newseed-20261029/seed{4,5,6}-all-candidates-12episodes.json`. The out-of-seed scalar gate is **deterministically replayed** from the previously recorded preprobe predictions and subsequent measured gains; it is **not a live-run demonstration of skipping GPU/CPU work**. All original source model forwards were actually executed to make evaluation unbiased.

## Results: cautious abstention but no forecast breakthrough

Mean of three per-seed ratios of online-shrunk prospective MSE to always predicting zero gain; **lower is better**, baseline = 1.000.

| Predictor | Mean MSE ratio vs zero | Scored episode abstentions | Proposed first probes |
| --- | ---: | ---: | ---: |
| Direction-only historical attention + new shrinkage | **1.00656** | 33/33 | 0 |
| Activation-conditioned attention + new shrinkage | **1.00704** | 33/33 | 0 |
| Online-calibrated activation attention + new shrinkage | **1.00596** | 33/33 | 0 |

| New seed | Direction ratio | Activation ratio | Calibrated ratio |
| --- | ---: | ---: | ---: |
| 4 | 1.01570 | 1.01427 | 1.01139 |
| 5 | 1.00034 | 1.00058 | 1.00042 |
| 6 | 1.00363 | 1.00627 | 1.00606 |

The analytic alpha often collapsed toward zero: final seed-wise shrinkage was **0.0000** for all heads in seeds 4 and 5, and **0.0120 / 0.0000 / 0.0079** for direction/activation/calibrated in seed 6. This prevents wildly large gain forecasts from sparse memory, but offers no independent evidence of useful fine-tuning improvements. In a truly enabled gating policy, no recommended probes means no forward-only learning actions: **safe abstention is not effective adaptation**. Hypothetical compute savings are not measured because every candidate was actually evaluated.

The admission summary under `research/evidence/lfm3-dust-newseed-20261029/seed456-shrinkage-admission-summary.json` is `hold_not_qualified`. All three policies fail a 5%-better-than-zero forecast requirement on all new seeds and abstain on every scored episode. No model-weight/controller-policy promotion is permitted.

## Verification, critical limits, and follow-up

- `src/lfm3_dust_prequential_audit.py` now accepts *only* the registered original seed 1–3/8-episode protocol or the preregistered new seed 4–6/12-episode protocol. Original experiment receipts still validate identically.
- `src/lfm3_dust_prequential_review.py` checks exact forward counts, new seed-specific task hashes, finite forecasts, signed direction pairs, 32-observation bounded causal attention memory, preprobe feedback timing and recalculated prequential forecast metrics.
- `src/lfm3_dust_uncertainty_replay.py` uses only those source receipts and past-episode training outcomes; `src/lfm3_dust_uncertainty_report.py` recomputes the compact results and refuses tampering or seed substitutions. Source execution receipts remain **unsigned local telemetry** and do not prove cryptographic custody, zero backward calls independently or the prompt-only provenance under malicious code.
- Local pytest before PR: **51/51** focused tests passing, including adversarial evidence changes; a separate dedicated hosted CI workflow is provided. The unchanged full repository **69% coverage gate** and Xwing admission/recovery controls still apply.

**Next test to consider:** an actual small online-trained neural head, trained *without backpropagation* via forward-only coordinate updates or recursive least squares on a genuinely richer feature representation from frozen LFM3. Pre-register task-disjoint and model-disjoint test groups with prospective online calibration error, gain-per-unit of total compute, and controls including random, momentum, curvature-aware Dust, AdamW and Muon. A new head should **not be enabled to control LoRA updates unless it beats a trivial baseline on new real tasks**, and any proposal must first be checked via fresh probes, independent evaluation, rollback, and human-controlled checkpoint promotion.

No release, merge, production service change, sustained benchmark workload, or trained model-weight deployment is authorized by this work.
