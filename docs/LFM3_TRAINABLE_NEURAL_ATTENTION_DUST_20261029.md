# Actual trainable LFM3 predictive cross-attention — forward-only pilot (2026-10-29)

**Decision: HOLD / NOT QUALIFIED FOR AUTOMATIC OPTIMIZER CONTROL.** This report describes a real trained, external parametric attention head fed by frozen LFM3's prompt-only activation summary. It does **not** describe an attention layer inserted into LFM3, adapter weights trained by the head, independently attested production deployment or generalizable user-facing gains. The research work is draft and isolated.

## Why this is different

Previous forward-only Dust/K2 papers and research PRs #13/#22–24, LFM3 optimizer arm #86, external direction/activation attention #87, online two-head coefficient gate #88, three-way LoRA-B AdamW/Muon/Dust parameter-matched benchmark #89, prequential negative control #90, and uncertainty abstention #91 established a strong test framework. However, previously deployed research heads had **fixed Q/K attention weights** and could only adapt bounded external mixture scalars or history. The new arm genuinely **trains 195 Q/K cross-attention parameters** using no gradient/backpropagation.

## Built and executed

- Two scaled-dot-product attention heads, each with a **learnable 4×12 Q matrix**, **learnable 4×12 K matrix**, and learnable zero-value null-key logit. One learned mixing logit combines the heads: **195 scalar parameters total**.
- The 12-dimensional query/key input is the **8-D LoRA-B perturbation direction** concatenated with a **4-D normalized prompt-only LFM3 final-attention V-projection input summary**. Value memory consists solely of previously measured training-example forward gain scalars, capped at 32.
- For each training episode beginning at episode 2, train the head against only **the previous episode's training labels**, using earlier episodes as attention keys/values. Try two deterministic antithetic head-parameter directions (four head-only parameter proposals) and accept a direction only if it strictly reduces this *past-only* training objective and stays within a 0.6 L2 parameter trust radius. There is no `.backward()`, PyTorch optimizer or torch import in the head. Queries for the current episode are generated **after training and before all current candidate gains are measured**.
- The adapter model is a real official pinned BF16 `LiquidAI/LFM3-1.7B` with frozen backbone, using the same two final V-projector LoRA-B matrices (2,048 possible perturbation scalars) as the previous Dust experiments. **Every experimental adapter perturbation is exactly restored after the measurement; no model optimizer updates are retained.** The head's 195 coefficients learn, but LFM3 model weights do not.
- Preregistered new procedural task seeds **7/8/9**, twelve distinct training prompts *within each seed*. Across seeds, 36 episodes yielded **35 unique task-input identities** because one same-family operand tuple recurred on two seeds; this must not be exaggerated as 36 independent examples or a new task family. All three families remain parity, greater-than and sum. The held-out selection and final test remain unopened.
- For each seed, 12 episodes × 8 positive/negative signed directions = **96 real pretrained-model perturbed forwards**, plus 12 prompt-context forwards and 12 clean-loss forwards; across all seeds = **288 perturbed LFM3 forwards + 36 context forwards**, **150 head-only objective evaluations** (50/seed) and **60 accepted head-only search proposals** (20/seed). The latter are all *head* parameter changes; no adapter changes.
- All 8 signed candidate directions were scored before measurements and independently evaluated, independent of any learned policy's preferred candidates. Cold-start episode excluded: **88 causal prequential gain forecasts per policy per seed**, 264 in aggregate, on frozen train examples only. No adaptive controller trained on selection or locked final-test examples.

## Real observed forecast results

Per-seed mean squared gain-prediction error **relative to predicting zero** (1.000 baseline; lower is better):

| Predictor | Seed 7 | Seed 8 | Seed 9 | Mean of ratios |
| --- | ---: | ---: | ---: | ---: |
| Predict zero gain | 1.000 | 1.000 | 1.000 | 1.000 |
| Historical mean | 1.131 | 1.178 | 0.962 | 1.091 |
| Fixed direction attention | 1.109 | 1.215 | 1.036 | 1.120 |
| Fixed activation attention | 1.186 | 1.245 | 0.936 | 1.122 |
| Previous online-calibrated attention | 1.190 | 1.242 | 0.958 | 1.130 |
| **NEW 195-parameter forward-only trained cross-attention** | **1.036** | **1.042** | **0.845** | **0.974** |

Actual new-head absolute MSE: seed 7 **0.0009672759** vs zero **0.0009336465**; seed 8 **0.0014001831** vs zero **0.0013441266**; seed 9 **0.0014198890** vs zero **0.0016799159**. Seed 9 alone shows ~15.5% lower error; seeds 7 and 8 are **worse** than zero by ~3.6% and ~4.2%. The average relative 2.6% advantage is not statistically reliable with three seeds and cannot support generalization claims.

Prospective top-ranked candidate gain difference versus the zero-predictor's fixed first-direction tie-break: seed 7 **+0.008741**, seed 8 **−0.002556**, seed 9 **+0.004548**. These are hypothetical gain-if-tested comparisons from actual all-candidate forwards, not measured task-quality improvement after training. The seed 8 failure makes the original per-seed admission requirement fail.

### Key interpretation

The neural head has **learned query/key weights**, not merely a revised mixture: all three seeds have a changed neural-parameter hash, 20 accepted head-only bounded proposals and a training-history replay that reconstructs each of the 195 weights and all 88 prospective scores. This is genuinely a new derivative-free **neural attention-head training mechanism** with no backpropagation. However, the performance evidence is still inadequate for replacing cheaper methods, and none of the LFM3 base or LoRA-B weights were actually optimized in this pilot.

## Safety and evidence custody

Source: `src/lfm3_dust_trainable_attention.py` (no torch dependency, explicit two-head Q/K matrices), and `src/lfm3_dust_prequential_audit.py` with a new **seed7–9-only** protocol preserving all older seed1–6 evidence. The strict verifier `src/lfm3_dust_trainable_neural_review.py` first runs the original all-candidate/held-out partition reviewer, then **reconstructs all online Q/K updates and every neural prediction** using previously recorded signed directions, four scalar prompt activation summaries, and training-only measured gain labels. It also recomputes MSE and top-1 acquisition metrics instead of trusting report aggregates. `src/lfm3_dust_trainable_neural_admission.py` requires **≥5% lower forecast MSE than zero AND positive top-1 candidate gain on every seed**, plus observed head-only training, before *nomination to a separate prospective study*. All results are **`hold_not_qualified`**. Even a future pass would never authorize live optimizer switches, model updates or checkpoint promotion.

Committed source receipts: `research/evidence/lfm3-dust-trainable-head-20261029/seed{7,8,9}-neural-head-12episodes.json`.

The local model-run receipts are **not independently signed**; reproducing a deterministic head's math does not prove that a malicious source couldn't forge all of its events. The historical Dust shared-signing-principal custody issue remains unresolved. The procedure uses toy public examples, 12 prompts per seed, an 8-D adapter perturbation subspace and very small expected gains; no new real-task generalization, compute/energy savings, robust confidence interval or autonomous recursive optimization is shown. Do not attempt a model/head promotion or merge while the separate repository-wide 69% coverage threshold and independent safety gates remain outstanding.

## Next research milestone

Conduct a larger and genuinely **task-family-disjoint** experiment, including real-language instruction-following, exact output correctness, held-out synthetic tasks, forgetting/retention and capability/safety diagnostics. Compare the 195-parameter head against zero, historical mean, random, momentum, curvature-aware Dust, AdamW and Muon at **matched total model-forward/backward equivalents, walltime and energy**, with fresh independent receipts. If forecast improvements become repeatable, place the bounded neural head in a **shadow recommendation harness** (no automatic updates), then consider a separately approved forward-only LoRA adapter training study that verifies post-update actual task gains, rollback, and correct separation of future test data. Keep explicit human-controlled promotion and Xwing recovery gates.
