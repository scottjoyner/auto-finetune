# LFM3 adaptive online attention gate — forward-only research, 2026-10-29

**Status:** draft isolated CPU-only pilot. No checkpoint writes, model-promoted adapters, production optimizer changes, release authority or loop daemon.

## Why this exists

Original Dust/K2 (draft PRs #13 and #22–24) established forward-only direction probes and a prior predictive direction classifier, but its classifier did not beat curvature-aware momentum on frozen synthetic confirmation and its custody boundary remained unresolved. LFM3 PR #86 built a causal direction-only head; PR #87 added **prompt-only internal attention activation summaries** and two deterministic attention views, without improvement in selection accuracy. This branch tests an explicit **online-learned two-coefficient gate** on top of the same LFM3 activation-conditioned architecture.

The distinction matters: this is a **trainable external predictive gate** (two coefficients) updated analytically using an exponential-weights / online expert-loss rule, **not a new transformer attention layer trained inside LFM3**. It receives an 8-dimensional candidate direction and a 4-scalar aggregate of prompt-only final-attention activation input; it compares two head predictions and updates their relative mixing weight according to later independently measured training-example forward-only gain. No backward calls or optimizer objects are used to train LFM3 or the gate.

## Causal feedback protocol

1. **At each training episode**, capture four scalar LFM3 prompt-only attention inputs and discard raw activations; generate four seeded perturbation directions.
2. Calculate both attention heads' predicted gains using only **prior training episodes**. Mix by gate weights and rank candidates, then freeze the top two selected directions for that episode.
3. Run positive and negative forward probes on the current public procedural training example. From clean-versus-perturbed loss, update the **future-facing** gate via clipped squared errors, with strictly positive minimum mixing weight of 5% per head. Only episodes after the current one may use newly observed direction labels for ranking.
4. Accept LoRA-B parameter changes only if freshly measured training loss improves within the 0.15 cumulative trust radius. Otherwise restore previous state. The gate cannot bypass the probe, act on model weights directly or use held-out evaluations as labels.
5. Re-evaluate **selection** questions only after the experiment; do not open the locked final-test partition, export adapters, start live model loading, or release Xwing admission/recovery holds.

The gate's two trainable mixing coefficients are **non-model** state. The score memory and calibrator are bounded. Each step uses two of four candidate directions and 4 antithetic perturbed forwards, plus one extra prompt-only attention-context forward. The explicit experiment remains restricted to two last-layer `v_proj` LoRA-B tensors, totaling 2,048 parameters; full Q/V AdamW and factor-wise Muon are NOT directly comparable.

## Executed real LFM3-1.7B results on x1-370

Official pinned BF16 checkpoint, three deterministic seeds, three training steps each. Pretrained backbone frozen; no backpropagation reported; 36 antithetic perturbed forwards across the three runs, plus 9 context forwards and separate unperturbed/selection passes. Three per-run source records under `research/evidence/lfm3-dust-online-calibration-20261029/`.

| Seed | Gate weights initially | Gate weights after step 3 (direction/context) | Selection before | Selection after | Accepted bounded steps |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.50 / 0.50 | 0.444 / 0.556 | 5/9 | 5/9 | 3/3 |
| 2 | 0.50 / 0.50 | 0.373 / 0.627 | 7/9 | 7/9 | 2/3 |
| 3 | 0.50 / 0.50 | 0.137 / 0.863 | 6/9 | 6/9 | 3/3 |

The coefficients **did learn from measured historical prediction errors**, so this is a concrete, observed feedback-driven calibration rather than a fixed-coefficient controller. However, the model's selection accuracy **never changed**, and the small training episode count is inadequate to establish generalization or optimizer superiority. Existing direction-only, activation-only and random baselines share the same initial frozen adapter and public dataset hashes, candidate population and antithetic probe count; the activation variants have additional full prompt forwards. A paired base report check also confirms the scores and budget.

**Training-loss acceptance is not task improvement** and does not prove beneficial self-modification. A gate coefficient shift is not evidence of superior prediction, and the observed increase in context-head weight might be overfitting to a handful of noisy gains.

## Review, CI, and restrictions

- Implementation: `src/lfm3_dust_online_calibration.py`, optional runner policy `activation_calibrated` in `src/lfm3_dust_predictive_probe.py`.
- Structural report gate: `src/lfm3_dust_calibration_review.py` rejects noncausal/missing calibration events, malformed weights, missing prompt-only context evidence, mismatched original random control and any checkpoint promotion claim.
- Tests: `python -m pytest -o addopts= -q tests/test_lfm3_dust_online_calibration.py tests/test_lfm3_dust_calibration_review.py tests/test_lfm3_dust_activation_head.py tests/test_lfm3_dust_activation_review.py tests/test_lfm3_dust_predictive_probe.py tests/test_lfm3_dust_forward_review.py`.
- Reported no-backward execution and context SHA-256 are **not independently attested**. The historical Dust/K2 cross-node signing-key custody issue remains unresolved. Do not treat a source-authored JSON as a trusted independent witness.
- The repository-wide 69% coverage threshold, protected admission gates, original Dust and LFM3 research provenance and all production launch controls remain unchanged.

## What to investigate next

Pre-register an **independent multi-episode** test across different public-task families with a held-out set untouched by both training and controller calibration. Evaluate prequential *prediction error* and directional gain per unit of **total actual compute**, against direction-only, activation-only, random, momentum and curvature-aware baselines. If the online gate cannot beat those, prefer the simpler method. To make AdamW, Muon and Dust a scientifically fair three-way optimizer comparison, train the **same 2,048 LoRA-B parameters** with each method, at matched compute/time and source data, with identical initial weights. No automatic weight promotion or live recursive optimizer controller should precede those gates.
