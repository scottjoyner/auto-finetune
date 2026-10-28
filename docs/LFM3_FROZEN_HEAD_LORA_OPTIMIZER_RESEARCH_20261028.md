# LFM3 frozen-LM-head LoRA optimizer loop — October 28, 2026

**Branch:** auto-finetune draft PR #61. **Status:** verified synthetic research only, not approved training on source traces or production deployment.

## What changed

1. `src/lfm3_lm_head_lora_lab.py` loads official `LiquidAI/LFM3-1.7B` BF16 revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. All **1,728,801,536 pretrained parameters**, including the LM-head base matrix, remain frozen. Final hidden activations are extracted in inference mode. A true rank-4 additive LoRA **on the LM-head projection only** trains new A/B matrices, `W_frozen + (alpha / rank) B@A`. No end-to-end model/backbone backpropagation, no attention adapter, no SFT or RL.
2. The official 128,000-token vocabulary gives **520,192 new trainable LoRA parameters** (`4*(2048+128000)`). Two **fabricated train** and two **separate fabricated dev** prompts use one-token A/B targets. Eight optimizer steps per arm, same initialization, source, labels and gradient clipping.
3. Native AdamW runs without special authorization. Muon is separately gated behind `--experimental-muon-lora-exception`, restricted to **these two newly constructed low-rank matrices**, never LFM3 base weights. The new `--compare-muon-rms-scaling` requires that exception and compares PyTorch's `adjust_lr_fn="original"` to `"match_rms_adamw"`. Old four-arm receipts remain unchanged.
4. `src/lfm3_lm_head_lora_observer.py` and `src/lfm3_muon_scale_observer.py` reload the pinned model read-only and score only **prespecified single-token optimizer option labels** in two predetermined prompt formats. All input fields, source flags, loss finiteness and option budgets are independently validated. A deterministic argmin computes the actual lowest dev loss; advisor output can only nominate a **new synthetic experiment** if both prompt choices agree with the objective.
5. `src/lfm3_optimizer_resume_lab.py` proves **bit-exact AdamW and Muon+AdamW toy-model restart**, interrupting after step five of 12 and restoring in-memory model/optimizer/RNG state. A caller-held SHA-256 must match before state deserialization, and parameter-group identity/shape/dtype are checked. It is **not** production durable checkpointing or independent signed source custody.

## Actual xwing BF16 optimizer results

One tiny dataset, one seed, eight steps per arm, CPU execution in an externally enforced network-disabled bubblewrap sandbox. Lower development cross-entropy is better.

| Method | LR 0.001 | LR 0.002 |
|---|---:|---:|
| AdamW | 11.674006 | 10.901610 |
| Muon original scaling | 11.801153 | 11.485044 |
| **Muon match_rms_adamw scaling** | **11.609944** | **10.669895** |

The RMS-matched Muon `LR=0.002` result is **0.231715 lower in dev loss than AdamW LR=0.002 on this tiny fixed task**. This is an observed mechanics result, **not** evidence Muon generally outperforms AdamW on LFM3, LoRA, or deployment. The comparison is underpowered, uncalibrated beyond these two rates, has no heldout and has not been independently repeated.

## Actual advisor outcomes

- **Four-option original comparison:** LFM3 chose `C` (Muon original LR 0.001) in both prompt variants, while external lower-loss option was `B` (AdamW LR 0.002). The controller refused.
- **Six-option calibrated comparison:** External lower-loss choice was `F` (Muon RMS-matched LR 0.002). LFM3 selected `B` in JSON presentation and `F` in compact table. The disagreement triggered a refusal to admit a next trial.
- In all cases: **zero advisor-triggered optimizer steps**, no persisted adapter, no modified pretrained model, no production routing, no source trace ingestion, and no autonomous recursion.

Exact source and receipts:
- `research/results/lfm3_real_frozen_lm_head_lora_adamw_muon_xwing_20261028.json`
- `research/results/lfm3_real_lm_head_lora_two_prompt_observer_refused_xwing_20261028.json`
- `research/results/lfm3_real_lora_muon_scaling_six_arm_xwing_20261028.json`
- `research/results/lfm3_muon_rms_scaling_six_choice_observer_xwing_20261028.json`
- `research/results/lfm3_optimizer_adamw_muon_inmemory_exact_resume_xwing_20261028.json`

## Risks and next release gates

- **No real-world task generalization claim:** two fabricated dev samples are far too few. Expand consented/independently sourced task corpus, separate dev and final heldout by session, audit split fingerprint provenance; do not optimize on repeated heldout feedback.
- **No semantic authorization:** no independently signed monotonic provenance witness; SHA-256 equality of untrusted receipts does not establish custody, consent, or production eligibility.
- **No autonomous optimizer control:** LFM3 recommendations were not reliable across prompt formats. Fixed policy owns all budgets, loss evaluators, source restrictions, weight promotion and rollback.
- **No durable crash-resume claim:** resume tests use in-memory bytes on one host. Need authenticated on-disk optimizer-state identity, crash/restart replay, corruption and partial-write testing before fleet operations.
- **Muon tensor routing:** low-rank matrices are experimentally reviewed only, not generally Muon-eligible. Dense attention/feed-forward Muon candidates still require separate shape scaling and multi-seed tests.
- **Hosted CI:** keep enforced 69% coverage floor; do not weaken thresholds or merge drafts because focused tests pass.

Companion SensorGraft PR #7 independently parses the first four-arm LoRA receipts and observer veto into a 6-node/7-edge **categorical** graph, with source authentication, training and production flags false. Do not promote that graph into a trusted semantic memory or head-plan generator.
