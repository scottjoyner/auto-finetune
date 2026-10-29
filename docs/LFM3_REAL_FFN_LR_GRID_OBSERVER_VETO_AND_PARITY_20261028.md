# LFM3 real FFN optimizer calibration: numeric gate overrides model advice

**October 28, 2026 | Draft research only | no production/self-modifying optimizer authority**

## Fixed experiment, pinned source and actual model training

This phase extends the previous three-seed AdamW-vs-Muon final-FFN LoRA comparison on the pinned official BF16 `LiquidAI/LFM3-1.7B` checkpoint revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. All **1,728,801,536 original pretrained parameters stayed frozen**. Real logits-to-final-FFN backpropagation updated only a temporary rank-4 LoRA hook at `model.layers.23.feed_forward.w2`, **38,912 trainable adapter parameters**. No adapters or base weights were saved.

The research code committed a **fixed 3 seeds × 5 optimizer/LR settings × 3 optimizer steps = 15 actual training trials** before execution on xwing under a network-disabled bubblewrap, pinned CPU runtime. Trained on **two fabricated codebook examples** and measured only **two fabricated development examples**. Training identity, paired seeds, optimizer configuration and exact earlier LR-0.002 development results are independently checked by `src/lfm3_ffn_optimizer_lr_gate.py`. The model did not select the five arms or alter any learning rate during these runs.

| Frozen final FFN LoRA optimizer | LR | Development loss (three seeds) | Mean |
|---|---:|---|---:|
| AdamW | 0.001 | 11.720094, 11.751378, 11.747183 | 11.739552 |
| AdamW | 0.002 | 11.107571, 11.227321, 11.210147 | 11.181679 |
| AdamW | 0.004 | **8.701119, 9.079185, 9.009270** | **8.929858** |
| Native Muon `match_rms_adamw` | 0.002 | 11.945520, 11.945520, 11.945520 | 11.945520 |
| Native Muon `match_rms_adamw` | 0.004 | 11.891962, 11.876373, 11.876313 | 11.881550 |

**Development-set winner: AdamW 0.004 (option C), all three adapter seeds**. This is a post-training development calibration on an already familiar tiny fictional task; it **does not demonstrate an independent model improvement** and no production optimizer configuration was changed.

Full evidence: `research/results/lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json`. Frozen code: `src/lfm3_ffn_optimizer_lr_calibration.py`. Independent numeric gate and adversarial tests: `src/lfm3_ffn_optimizer_lr_gate.py`, `tests/test_lfm3_ffn_optimizer_lr_gate.py`.

## Critical self-observer reliability result: two incorrect but consistent LFM3 votes

The unchanged pretrained LFM3 model read the **externally verified numeric loss table only**, in two fixed prompt formats, through a separate inference-only process. A choice was computed by next-token logits over fixed A/B/C/D/E options. **It chose A = AdamW 0.001 in BOTH layouts**, even though the independent numeric gate selected C = AdamW 0.004 consistently across all seeds.

This is a documented **self-observation failure**: LFM3 was consistent with itself, but contradicted the objective measurements. The external controller **vetoed** any optimizer research proposal (`dev_only_review_candidate:null`) and did not change a hyperparameter, weight, evaluation threshold, trace, model selector, or routing policy. It must remain possible to reject confident/consistent LFM3 advice; prompt agreement is NOT correctness.

Evidence: `research/results/lfm3_real_ffn_five_arm_lr_original_model_veto_xwing_20261028.json`. Observer implementation: `src/lfm3_ffn_optimizer_lr_observer.py`; adversarial choice/logit tests: `tests/test_lfm3_ffn_optimizer_lr_observer.py`.

## Pre-registered one-shot different fictional prompt-family parity audit

Without consulting LFM3's rejected recommendation, the independent development gate had already chosen LR 0.004. A new, **frozen four-example fictional parity task** (even/odd integer classification to the same A/B output tokens) was created and committed before the first read. The original two train and two dev prompts were disjoint from all parity prompts. This task family differs from the codebook prompts, but **is not a trusted, independently curated broad real-world benchmark**; it is still only four fabricated A/B examples.

Exactly two fixed AdamW arms per seed, LR 0.002 (earlier baseline) and LR 0.004 (development-selected candidate), retrained three gradient steps. Each reproduced the previous development cross-entropy before a **single parity read-only evaluation**. No further training followed holdout access. Frozen original pretrained parity loss: **5.023702**.

| Seed | AdamW .002 parity CE | AdamW .004 parity CE |
|---|---:|---:|
| 20261028 | 4.399837 | **3.056314** |
| 20261029 | 4.481154 | **3.344643** |
| 20261030 | 4.455047 | **3.218110** |

The observed parity difference favors AdamW .004 in all three fixed seeds, but it is still only a 4-example same-token small synthetic transfer test, not evidence of general capability improvement or a legitimate production learning-rate change. **This parity holdout is now permanently spent for optimizer/learning-rate selection**: subsequent LR tuning must use fresh, independently curated tasks, not recycle it.

Evidence: `research/results/lfm3_real_ffn_adamw_lr002_vs_004_parity_heldout_xwing_20261028.json`. Independent review in `src/lfm3_ffn_parity_transfer_gate.py` validates source/task checksum, precommitted LR choice, six paired actual-model training arms, exact dev replay, nonfinite/forged losses, frozen base, false source/promote claims, and `parity_holdout_spent_for_optimizer_selection:true`. Additional tests reject incorrect seed, optimizer, source, tampered scores, or post-holdout updates.

## SensorGraft independent categorical audit

The companion SensorGraft draft PR #7 imports *only the metadata receipts*, never pretrained/adapter weights, into three bounded fictional research fixtures. `src/lfm3_ffn_lr_reflection_veto.rs` independently recomputes all fifteen real development losses, the five means, per-seed numeric best, rejects incorrect source/trust claims, checks actual next-token logit argmax against both LFM3 A votes, verifies the parity trial's independent dev continuity and non-authoritative spent holdout, and creates **9 symbolic nodes and 8 typed edges**. The new strict scratch journal CLI replays at revision 2. No SensorGraft production service or Neo4j/NAS data was modified.

## Still required for responsible closed-loop research

1. **Curated, independently sourced multi-family holdouts** with no exposure during optimizer search; task-specific model quality, not just loss on four chosen labels.
2. Reproducible multi-seed/multi-node ROCm results and bounded compute budgets, including AdamW vs Muon parameter-shape-specific behavior and native crash/rollback.
3. Signed, independent source witnesses and authenticated human model promotion enforced **at the model-serving boundary**, not an LLM-generated confirmation bit.
4. Keep existing hosted Python **69% coverage** and SensorGraft journal-concurrency **issue #8** as blockers, with issue #9 source custody unresolved.
5. Treat any original-model optimizer suggestion as **untrusted advice**, checked by an immutable numeric evaluator. A veto must not silently trigger more retries on spent holdouts.

This is measured progress in **controlled optimizer research**, not verified recursive self-improvement.
