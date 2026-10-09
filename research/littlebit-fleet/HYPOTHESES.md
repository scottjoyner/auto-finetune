# Preregistration: hypotheses, endpoints and inference discipline

**Research date:** 2026-10-09. **This is a prospective plan; no LittleBit-Fleet observation is asserted.** The plan is locked at the first E1 measured result. Changes must have a versioned amendment and identify which data/results were already observed.

## Target estimands and common design

The unit of comparison is a **paired run** using the same frozen teacher/model revision, tokenizer, nonoverlapping data manifests, sequence length, quantized layer mask, effective bit budget, optimizer updates or forward-evaluation budget, and seed where applicable. Compare within hardware/kernel and report separately for each model and BPW. Record absolute differences and relative changes; do not infer equivalence from failure to reject a null.

**Primary quality endpoint:** heldout token-weighted cross-entropy (CE), not training loss, from a fixed precommitted evaluation set. **Secondary:** per-task agentic outcomes, argument-schema correctness, refusal/error robustness, PPL=exp(CE), task/format adherence, model bytes, end-to-end throughput and training costs.

## H1 — Better initialization

- *Null H1_0:* On fixed matrices and equal rank/BPW, Joint-ITQ does not outperform Dual-SVID on reconstruction error averaged over predeclared seeds; after matched QAT, no heldout CE improvement is established.
- *H1_A:* Joint-ITQ improves signed-factor alignment/reconstruction and/or downstream heldout CE without extra deployed weight cost.
- *Treatment:* original SVD/Dual-SVID vs upstream Joint-ITQ enabled via `--use_itq True`.
- *E1 primary:* relative Frobenius reconstruction error (smaller better); output cosine and independent layer activation MSE are secondary.
- *E2/E3 primary:* heldout CE after equal optimizer steps; report pre-QAT CE separately to distinguish initialization from adaptation.

## H2 — Scales-only QAT

- *Null H2_0:* Freezing latent factor sign/master values and optimizing only `h,g,ell` (and residual equivalents) does not improve heldout CE over an untrained initialization at fixed BPW.
- *H2_A:* Scales-only QAT recovers a measurable fraction of the heldout CE lost to compression while reducing **peak training memory and/or per-step wall time** relative to full QAT.
- *Controls:* untrained initialization, upstream full QAT, matched scales-only QAT, and optionally latent-scale-only ablation.
- *Compute fairness:* match update count AND present per-token cost; if forwards/backwards differ, also report a matched wall-time comparison.
- *Decision:* E2 advances only if scales-only heldout CE is lower than no-QAT on the predeclared smoke (no tuning on heldout); assess multi-seed repeatability before claiming an improvement. A training-side decrease alone is not success.

## H3 — Forward-only scale updates

- *Null H3_0:* At a matched **model forward evaluation** budget, structured orthogonal antithetic scale estimates do not approach the exact local autograd gradient or heldout improvements better than matched random/gaussian alternatives.
- *H3_A:* Orthogonal antithetic directions yield lower estimation variance and useful heldout optimization relative to a random-direction alternative at an acceptable inference-time/computation cost.
- *Arms:* exact-local-autograd scales-only, orthogonal antithetic scales-only, independent gaussian antithetic scales-only, zero-update, and random-shuffled/sign-flipped gradient negative control.
- *Calibration metrics:* cosine similarity to exact local gradient, relative L2 error, across-seed variance, loss-improvement direction, forward count, elapsed time, peak memory and update cosine.
- *Reference:* PR #13 calibrated K=1024, sigma=.25, D=4 on **K2's final o_proj adapter**. These are a starting reference, NOT prespecified optimal hyperparameters in the scale parameter space.
- *Primary decision:* no promotion on cosine alone. Require heldout CE improvement, no repeated wrong-sign descent, and acceptable cost **before** an 8/16/32 update schedule.

## H4 — Useful compression / operational viability

- *Null H4_0:* Claimed block BPW savings do not yield a materially useful whole-checkpoint and quality/latency operating point relative to same-node FP16/BF16 and Q4 controls.
- *H4_A:* At 0.55 BPW (explore 0.30 only after E2), the model attains a useful storage-versus-quality tradeoff and deployable deterministic inference without unacceptable throughput regression.
- *Primary storage metric:* measured total on-disk checkpoint+metadata+tokenizer bytes; also report transformer-block effective BPW on a separately stated parameter denominator.
- *Independent operational endpoint:* same-node same-evaluation quality and latency comparison. No single number conflating block BPW and whole-model bits/weight.

## Fixed acceptance discipline

1. Run E0/E1 CPU synthetic across seeds **7, 42, 1337**, spectra {low-rank, flat, heavy-tailed} and shapes chosen before first measurement; record rank/bit-budget calculations.
2. E2: 0.55 BPW only, one verified upstream-compatible public teacher (proposed Qwen3-0.6B), at most **100 training steps in the first smoke**. Train split distinct from fixed heldout and tests.
3. Gate from E2 to E3: at least **one preregistered scales-only arm** must reduce CE versus its no-QAT baseline on heldout; otherwise **HOLD** and report negative pilot. Only then use 3 predeclared seeds and independent evaluator. Do not tune against test split.
4. Report uncertainty (paired per-seed deltas and dispersion, paired bootstrap over independent documents/tasks when suitable). Three seeds are exploratory, not statistical proof; any confirmatory superiority threshold/sample size must be preregistered **before** confirmatory run.
5. Report all runs, stopped runs, divergence/OOMs, CPU/GPU fallback and invalidated splits. Publish missing results as `NOT_RUN` rather than fabricated zeros.
6. The final report contains predictions in an immutable dated table and observations in a separately appended table. No goalpost movement after examining outcomes.

## Operational stopping rules

Hard-stop on license hold, unsupported model architecture, data-lineage mismatch, model/checkpoint hash change, unexpected network/provider calls, VRAM/RAM floor, unowned GPU, uncontrolled NAS path, unexpected background process, invalid bit accounting, nonfinite training loss, quantized serialization mismatch or accidental production side effects.

No assertion that 0.30 or 0.10 BPW can preserve useful reasoning on the target fleet is authorized by the source paper alone.
