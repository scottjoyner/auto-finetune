# LFM3 predictive Dust: training-only attention task-shift safeguard — October 30, 2026

**Research outcome: novelty guard observable; optimizer quality NOT established.** The third LFM3 optimizer arm remains backprop-free predictive Dust, alongside AdamW and experimental Muon. This branch adds a *train-only* task-shift detector using actual frozen LFM3 attention features, verifies it on a new public instruction family, and makes the shared three-method registry require the new evidence before recognizing Dust as an inspectable method. **No learned head or model-update policy was activated.**

## Why we need task-shift detection

Draft #92 trained 195 parameters in an external two-head Q/K attention predictor using bounded forward-only head-parameter proposals. The predictor observes compressed prompt-only LFM3 attention information and prior, measured LoRA-B perturbation gains. Draft #93 tested learned head updates on first/last word-extraction training, with head and attention memory frozen on previously unseen uppercase rewriting evaluation. Despite legitimate training, the learned head's prospective gain forecast MSE was **1.116 / 1.171 / 1.126 times worse than predicting zero gain** by seed. The head is operational but poorly calibrated on unfamiliar instructions, so permitting it to autonomously choose model updates would be unsupported.

## Frozen guard trained ONLY on historical source training episodes

For each previously trained neural head (source seeds 10–12), the new `FrozenTaskShiftGate` uses the **12 extraction-family training prompt contexts** only. Each context is a four-dimensional normalized scalar summary of the final LFM3 attention V-projection input. A novelty threshold is computed using the minimum training leave-one-out nearest-neighbor cosine similarity, minus a **preregistered 0.02 cosine margin**. Crucially, none of the historical uppercase transfer contexts or their measured gains are consulted when fitting the threshold.

A separate skill gate recomputes **80 prequential gain forecasts per source seed** (training episodes 2–11 × eight signed perturbations), compares the trained neural head's historical MSE with the trivial zero-gain forecast and insists on at least **5% lower error** before allowing any proposed update. Original training neural/zero error ratios were **1.0115 / 0.9842 / 1.0544**, so every source head fails the 5% skill criterion. A model update would be refused even if a novel prompt happened to look in-distribution.

For each new prompt, the guard computes the nearest training-context cosine and sets `abstained=true` if (a) the prompt is outside the train envelope or (b) the head lacks previous train-only forecast skill. When it abstains, it replaces the output forecast with **zero** and never schedules a probe. This is a **conservative heuristic**, not a calibrated OOD probability, formal confidence bound or established way to improve task accuracy.

## Fresh pinned LFM3 experiment: unseen alphabetization instructions

We preregistered seeds **13, 14, 15** for six new public instructions each: sort two lowercase words alphabetically. These were **not** used to train the earlier 195-weight heads (trained on extraction) or select their uncertainty thresholds. The only new pretrained-model execution was the **prompt-only context forward** of real pinned BF16 `LiquidAI/LFM3-1.7B` for each new instruction, offline on x1-370, no model perturbation or gradient run. The trained Q/K head was reconstructed from its historical train-only records; its 195 weights and 32-entry memory were frozen. It forecast eight signed candidate directions per prompt, but the guard never requested or executed those perturbations.

| New seed | Paired training head seed | Novel task contexts flagged | Candidate forecasts withheld | Model perturbation forwards |
| --- | --- | --- | --- | --- |
| 13 | 10 | **6/6** | 48/48 | 0 |
| 14 | 11 | **6/6** | 48/48 | 0 |
| 15 | 12 | **5/6** | 48/48 | 0 |
| **Total** | — | **17/18** | **144/144** | **0** |

The remaining one novel sorting prompt on seed 15 had enough raw activation similarity to pass the task-shift threshold. This is a **real false negative** of novelty detection, not a successful generalization result. That prompt was *still blocked* by the independent historical skill gate. Importantly, the new novelty experiment collected **zero candidate-gain labels** and thus cannot measure or claim new-task predictive MSE, post-update accuracy, saved training FLOPs or how often useful candidate updates were rejected. Only **18 real pretrained prompt-only context forwards** were executed.

The reports claim zero backward calls, zero retained model LoRA-B updates and exactly matched before/after adapter SHA256. These are local source-authored execution receipts, not independent cryptographic attestations.

## New code, strict review, and integrated third-method registry

- `src/lfm3_dust_task_shift_gate.py`: frozen train-only LFM3 activation nearest-neighbor novelty detector and 5%-better-than-zero prequential skill gate. No gradient, torch dependency or model weight update.
- `src/lfm3_dust_task_shift_probe.py`: offline, explicit opt-in, real pinned pretrained model prompt-only forward on a new instruction family with zero perturbation or training attempts.
- `src/lfm3_dust_task_shift_review.py`: rebuilds the 195 trainable Q/K weights from the original training-only source, recomputes the novelty threshold, all 18 prompt/direction identities, raw neural predictions and every abstention decision; fails closed on changed trained-head evidence, altered context, candidate omission or falsely claimed model updates.
- `src/lfm3_dust_task_shift_admission.py`: cross-seed accounting for **17 of 18 novel contexts flagged** and **all 144 candidate forecasts withheld**. No independent confidence guarantee and no optimizer authority.
- `src/lfm3_three_method_shadow_registry.py`: the common AdamW/Muon/Dust evidence inventory now requires **15 validated model-evidence files**, three additional real new-family reports alongside the prior matched optimizer and head-training evidence. The third predictive Dust method displays the new novelty guard's `shadow_guard_validated_not_optimizer_qualified` status. Any missing or falsified receipt blocks the interface, including if it claims positive evidence without a separately reviewed new admission design.

All three records are under `research/evidence/lfm3-dust-task-shift-20261030/seed{13,14,15}-sort-two-words-shadow.json`.

### Empirical limits and next experiment

The 36 historical train examples were used to set cosine thresholds. Their in-envelope results are therefore not an independent estimate of false alarms. Only 18 novel public prompts from **one new procedural task family** were assessed, and one was missed. The entire pipeline still lacks independently controlled task-quality evidence, signed source execution custody and compute-matched AdamW/Muon/Dust comparisons. This guard is designed to make costly mistakes less likely, not to claim strong model capability.

Next, preregister multiple genuinely different new instruction families with both in-family and out-of-family controls, prospective gain labels only in a review-only evaluation phase, and measurement of **gain per model-forward-equivalent and token**, not just direction forecasting. Include train-only calibration with confidence intervals over task families. Keep all trained adapters, optimizer switches and inference deployments under explicit human-controlled Xwing/coverage/recovery gates.

**No production changes, model checkpoint exports, automatic self-training, merge or coverage reductions were made.** This branch remains draft research stacked on PR #95. Existing **69%** full-repository coverage threshold remains in force.
