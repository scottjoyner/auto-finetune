# Actual LFM3-1.7B native AdamW/Muon mid-training LR telemetry and external model-vote gate

**2026-10-28 — xwing, research-only — draft PR #61.** This document records an executed experiment, not recursive self-improvement or a production activation.

## Model, isolation and frozen protocol

Official BF16 LiquidAI LFM3-1.7B checkpoint revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, original checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. All **1,728,801,536 pretrained parameters** remained frozen. Genuine model forward/backpropagation trained only a temporary rank-4 additive LoRA at `model.layers.23.feed_forward.w2` (**38,912 trainable adapter parameters**). No checkpoint/adapter archive was saved or deployed.

The previously committed research script `src/lfm3_ffn_fixed_lr_schedule.py` ran on **xwing** using CPU, network-disabled bubblewrap, pinned Transformers 5.21.0, native Torch AdamW and native Muon `adjust_lr_fn="match_rms_adamw"`. Training was three gradient steps per arm on **two fabricated codebook training examples**; two fabricated codebook development examples were observed after each step. No heldout data was consumed.

Precommitted design: **three frozen seeds** `20261028`, `20261029`, `20261030`; **four fixed schedules** per seed: AdamW or RMS-matched Muon at LR `[.004,.004,.004]` versus `[.002,.004,.004]`. Each schedule kept the same optimizer instance/state through its bounded three-step run. No LFM3-generated text, model output or review influenced the actual learning rates. All twelve trials completed with finite model gradients and losses and an untouched pretrained backbone.

| Native optimizer | Schedule | Seed 20261028 | Seed 20261029 | Seed 20261030 |
|---|---|---:|---:|---:|
| AdamW | fixed `.004,.004,.004` | **8.701119** | **9.079185** | **9.009270** |
| AdamW | `.002,.004,.004` | 9.218925 | 9.521599 | 9.458870 |
| Muon RMS-matched | fixed `.004,.004,.004` | **11.891962** | **11.876373** | **11.876313** |
| Muon RMS-matched | `.002,.004,.004` | 11.891973 | 11.891962 | 11.891960 |

All four experiment variants used actual native optimizer state. The fixed `.004` schedules had lower fictional *development* loss in all three paired seeds than the precommitted switch after step 1. The two fixed `.004` arms also reproduced the prior independent 15-arm real model development calibration to a 0.01 tolerance. This result **does not prove that LR changes generally hurt**, only that this specific first-step change did not help on a tiny synthetic protocol.

Raw, review-only, complete per-step evidence: `research/results/lfm3_real_ffn_native_fixed_vs_step1_lr_switch_12trials_xwing_20261028.json`. `src/lfm3_ffn_fixed_lr_schedule_gate.py` independently checks all 12 ordered arms, 36 step measurements, exact optimizer/LR schedule, gradient validity, pinned checkpoint and source revision, original three seeds, prior calibration continuity and false execution/production flags. `tests/test_lfm3_ffn_fixed_lr_schedule_gate.py` covers forged LR, gradient, seed/baseline, metric and claimed authority.

## Actual unmodified original LFM3 model observing the telemetry

Next, the **unaltered original pretrained LFM3** was loaded in a new read-only network-disabled bubblewrap process and given the externally verified stepwise telemetry. For each optimizer (AdamW/Muon) it viewed structured numeric data and a reordered text presentation. Fixed options: `A` = LR `.004,.004,.004`, `B` = LR `.002,.004,.004`. The output choice was measured by **real one-token next-token logits**, never passed as a command.

| Original-model observation | Choice | External numeric truth |
|---|---|---|
| AdamW structured | A | A |
| AdamW reordered | A | A |
| Native Muon structured | A | A |
| Native Muon reordered | A | A |

All **four** actual model votes matched the independently checked fixed-schedule winner. Unlike the earlier **five-option** original LFM3 experiment where it voted **A/A for LR .001** but the external best was **C = AdamW LR .004**, this reduced two-option comparison succeeded. The earlier failure remains material evidence of format/task dependence and prevents any claim that the observer can generally pick correct optimizer settings.

`src/lfm3_ffn_schedule_self_observer.py` enforces bounded, typed, nonexecuting views and two fixed layouts per optimizer; `independent_judge()` requires full external gate identity and all-source **no training** flags, never trusting model logits as a source of metric truth. Adversarial tests deny forged external authority, logs and optimizer changes. The real observer was rerun after strengthening its numeric gate.

Final source `db06a8b35b504f55fc6fc27a7f6f1af9ec82b628` real observation is in `research/results/lfm3_real_ffn_stepwise_lr_observer_four_correct_xwing_20261028.json`.

**Proposal produced:** `{"optimizer":"adamw","learning_rates":[0.004,0.004,0.004],"execute":false,"human_review_required":true,"requires_new_independent_evaluation":true}`. No adapter created, optimizer configuration changed, new training started, automatic weight updates made, model promoted or production trace written.

## SensorGraft linked audit

The companion draft SensorGraft PR #7 contains the **12 actual native optimizer trial metadata packet** and **four real original-LFM3 vote receipts**, not neural weights. Strict Rust source checks produce a **49-node/48-edge per-step training graph**. A separate independent observer verifier joins four model next-token decisions and external numeric comparisons into a **53-node/52-edge** graph. A new isolated `/tmp/sg-lfm3-lab-*/` journal was actually created and cold-replayed at revision 2 with 53 nodes/52 edges. All units, full serial all-target Rust suite, formatting and strict Clippy passed.

## Open gates and next meaningful work

- **Do not reuse the previous fictional parity heldout for tuning.** The new schedule trial did not evaluate independent heldout examples. Next experiments need pre-registered, externally curated task families with genuine source custody, much more data and realistic performance metrics.
- Hard operator authorization at the serving boundary (not the registry's caller boolean), signed/monotonic source custody, replay/rollback, real failure recovery, and independent multi-node ROCm execution remain unverified.
- Hosted Python mandatory coverage **69%** remains a PR blocker; adding research code without exercising it in hosted CI may lower aggregate coverage. No threshold was weakened.
- SensorGraft journal parallel lock issue #8 remains OPEN. Serial test success **does not prove parallel locking correctness**; source-custody issue #9 remains OPEN.
- No model-driven optimizer execution, modification of production weights, fleet routing, NAS, Neo4j, production traces, or system cutover was performed.

**Conclusion:** an actual in-run LR transition and a bounded real model-observation feedback step were demonstrated. Neither self-directed model training, validated long-horizon optimizer adaptation, nor recursive self-improvement was demonstrated.
