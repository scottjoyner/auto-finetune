# LFM3 predictive Dust: trainable attention transfer to unseen instruction family
*Research protocol and real results — October 29, 2026*

**Decision: HOLD / NOT QUALIFIED.** The derivative-free trainable attention head can adapt 195 attention parameters from past measured forward-only gains, and it outperforms earlier fixed/online-mix heads on this procedural task transfer. It still produces **worse gain forecasts than simply predicting zero**, so no optimizer or model-weight automation is authorized.

## Background

The prior Dust/K2 work (draft PRs #13, #22–24) introduced backprop-free perturbation and predictive direction selection. LFM3 research PRs #86–91 developed a bounded last-two-V-projection LoRA-B third update arm, causal attention over past scalar gains, an activation-conditioned head, an online learned mixer, a parameter-matched AdamW/Muon/Dust comparison, and negative-control / abstention audits. Draft PR #92 proved that **a 195-parameter external two-head scaled-dot-product attention network can train its Q/K projections with no backpropagation**, using real LFM3 prompt-only activation features and train-only forward perturbation outcomes. The learned head was not yet reliable across seeds.

A new and more discriminating question is: **does a learned LFM3 predictive head transfer to a task family on which it received no labels?** This experiment tests precisely that, without automatically updating any LFM3 weights.

## Frozen, registered disjoint-family protocol

- Three registered procedural seeds **10, 11 and 12**; real pinned pretrained BF16 LiquidAI/LFM3-1.7B on x1-370, offline CPU-only. Strict pretrained guard, LoRA-B parameter scope 2,048 scalars across final two attention `v_proj` tensors, frozen backbone and Q/V adapters. All current-episode plus/minus perturbations restored via exact `copy_` after each measured forward; starting and final adapter SHA-256 match.
- Per seed: **12 training prompts** alternating between *first-word extraction* and *last-word extraction*, with explicit input/answer pairs and train-only outcome feedback. The neural head's 195 query/key/null/mixer weights are trained with forward-only 2 × antithetic head-only proposals and a bounded trust radius on *prior* training episodes; never use the current episode's labels while forecasting it.
- Then perform one final permitted **training-only head update** using the twelfth training episode's outcomes, freeze the head's weights and its 32-entry attention memory, and evaluate **six novel uppercase-rewriting prompts** without any extra head training or memory feedback. No uppercase examples or gains are supplied to the head. All six transfer episodes are prospectively forecast BEFORE probing.
- For every example, generate four deterministic low-dimensional candidate LoRA-B perturbations with both ± signs; **all eight** candidates are measured regardless of the policy's recommendation. Compare six forecast policies on identical outcomes: always predict zero, historical mean, direction-only attention, fixed activation attention, online-calibrated attention, and the new trainable 195-weight neural head.
- **36 distinct extraction training prompts** and **18 distinct uppercase transfer prompts** across three seeds; no reused prompt identities across the cohort. Same words and procedural generation can recur as raw vocabulary. These are task-family-disjoint synthetic instructions, **not a demonstration of generalization to arbitrary real-user tasks**.
- A total of **54 training/evaluation prompt episodes**, **432 real signed-perturbation LFM3 forward passes**, **54 prompt-only LFM3 context forwards**, clean control forwards, **160 head-only loss evaluations** and **64 accepted head-only bounded parameter proposals**. All LFM3 gradients, retained adapter updates, exports and production service changes remain zero. No model parameter training or promotion was performed; only the *external head's* weights trained.
- The original arithmetic seed1–9 selection and locked final-test examples were never evaluated or used for training. The new uppercase family is an **evaluation** family: its measured probe gains are used solely for scoring and are not available as feedback.

## Actual, newly recorded results

Relative MSE of **prospective gain prediction on the unseen uppercase family**, divided by the MSE of always predicting zero gain. Baseline=1; lower is better.

| Forecast policy | Seed 10 | Seed 11 | Seed 12 | Equal-seed mean |
| --- | ---: | ---: | ---: | ---: |
| Zero gain | 1.000 | 1.000 | 1.000 | 1.000 |
| Historical mean | 1.718 | 1.245 | 1.165 | 1.376 |
| Direction-only attention | 3.134 | 1.672 | 1.585 | 2.130 |
| Fixed activation-conditioned attention | 4.079 | 1.753 | 1.497 | 2.443 |
| Old online calibrated activation attention | 3.527 | 1.811 | 1.448 | 2.262 |
| **New trainable 195-weight cross-attention** | **1.116** | **1.171** | **1.126** | **1.138** |

Absolute neural-head transfer MSE: seed 10 **0.0001462521** versus zero **0.0001310156**; seed 11 **0.0001571836** versus **0.0001342046**; seed 12 **0.0001221662** versus **0.0001084577**.

Neural top-ranked signed-direction measured-gain advantage against zero-predictor's first-direction tie-break (per transfer episode mean): seed 10 **+0.006566**, seed 11 **−0.005964**, seed 12 **−0.006505**. These are **counterfactual candidate-choice comparisons**, not post-training held-out task accuracy improvements.

The head did meaningfully learn: 22 / 21 / 21 accepted bounded head-only proposals and 55 / 54 / 51 head-only objective evaluations for seeds 10 / 11 / 12. Head weights and memory hashes remained **exactly frozen** throughout the uppercase portion. Nevertheless, the neural head was **worse than zero on every seed**, and chose worse directions than the zero policy on two seeds. It is thus *not* permitted to self-tune LFM3 using these forecasts.

## Evidence provenance and enforcement

Raw real-run JSON evidence is committed under `research/evidence/lfm3-dust-text-transfer-20261029/seed{10,11,12}-extraction-to-uppercase-18episodes.json`. Source `src/lfm3_dust_text_family_curriculum.py` regenerates the original public task examples and SHA-256 partitions. `src/lfm3_dust_text_family_transfer_review.py` deterministically reconstructs the neural head's Q/K tuning episode by episode and every other head's forecasts, verifies each prompt-only attention summary hash, checks all ± perturbation coefficients, recomputes all future-family MSE and candidate-choice outcomes, and verifies **no transfer feedback entered any predictor**. `src/lfm3_dust_text_transfer_admission.py` checks three unique seeds, all reported independent example identities, and requires at least 5% lower error than zero on *each* seed and positive top-1 gain advantage on each before suggesting another separately evaluated study.

Actual result: **`hold_not_qualified`**. Independent reviewer source evidence is still local source-authored data, not cryptographically separate model-execution attestation. The historical Dust/K2 independent signing-custody weakness remains outstanding. Do not misrepresent mathematical receipt replay as proof against a malicious measurement agent.

## Implications and next experiment

The 195-parameter learned Q/K head achieves a major relative reduction in forecast error versus our prior learned attention predictors on an unseen procedural family. But with simple constant zero outscoring it, head-based adaptive optimization remains unjustified. This is a falsifiable negative result rather than a production regression.

Promising next research directions include **task-conditioned uncertainty estimates**, train-only gain distribution normalization, an explicit null/abstention head (learned on future-disjoint training tasks), and learned selection by *gain per total model forward equivalent*, against random/momentum and curvature-aware direction controls. The next dataset should include real text instructions and genuinely independently adjudicated outcomes, a larger number of task families, and train/validation/test held separately from controller feedback. Every new design needs a prospective independent cohort and real model training comparison before any autonomy gate can be reconsidered.

All work remains stacked draft research: Xwing admission and recovery gates, source provenance, reviewer sign-off and the unchanged **69% full-repository coverage** requirement apply. No automatic optimizer switch, weight promotion, model checkpoint, production inference routing, or uncontrolled persistent benchmarking is authorized.
