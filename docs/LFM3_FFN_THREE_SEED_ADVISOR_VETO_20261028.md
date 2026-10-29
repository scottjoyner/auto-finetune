# LFM3 real-FFN three-seed optimizer loop and prompt-sensitive self-observer — 2026-10-28

**Research-only; auto-finetune draft PR #61.** Executed on xwing with pinned, official BF16 LFM3-1.7B checkpoint revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. Network-disabled bubblewrap with read-only source/model mounts; CPU. Production models, Neo4j, NAS, routes and weights unchanged.

## Actual training

A **real graph-backpropagated** rank-4 LoRA hook on final LFM3 feed-forward projection `model.layers.23.feed_forward.w2`; **38,912 new trainable parameters**. All **1,728,801,536 pretrained parameters frozen**, no base/adapter weight checkpoint persisted. Three independent seeded LoRA initializations (`20261028`, `20261029`, `20261030`); AdamW vs native PyTorch Muon `adjust_lr_fn="match_rms_adamw"` at LR 0.002, three gradient steps per arm, six genuine model-forward/backward training runs. Two entirely fabricated train + two fabricated dev prompts, **no independent heldout**.

| Adapter seed | AdamW dev cross entropy | RMS-matched Muon dev cross entropy | AdamW relative advantage |
|---|---:|---:|---:|
| 20261028 | 11.107571 | 11.945520 | 7.01% |
| 20261029 | 11.227321 | 11.945520 | 6.01% |
| 20261030 | 11.210147 | 11.945520 | 6.16% |

All start from `11.945557` development loss. In this **final FFN internal-layer experiment**, AdamW beats RMS-matched Muon on all three seeds. This contrasts with the **different** frozen-output-head 18-arm experiment, where RMS-matched Muon had the lower dev loss for all three seeds. This is a falsifiable demonstration of **tensor-location and optimizer scaling sensitivity**, *not* generalizable LFM3 quality improvement. Reusing the same two-example dev split also limits inferential power.

Exact committed data: `research/results/lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json`.

## Deterministic research admission, not model-controlled training

`src/lfm3_ffn_multiseed_guard.py` independently checks exact model revision/checkpoint, fixed optimizer and LR per arm, three seed identities, 3-step budget, 38,912 trainable adapter count, finite paired losses/gradients, flags indicating original weight integrity and no checkpoint writes, and independently recomputes the relative per-seed dev losses. It refuses unknown fields, incorrect reported winner, cross-seed aliasing, nonfinite telemetry, or a single adverse seed. A **2% relative per-seed improvement floor** is required to *nominate a new fabricated research experiment for review*. It does not execute trials, change LR or choose trusted source data. Four new guard tests passed on xwing.

`src/lfm3_ffn_multiseed_observer.py` passes only the independently computed median dev losses to the original **unaltered** pinned LFM3 model, as a two-option A/B choice using two fixed prompt layouts. The actual LFM3 observations:
- **A**: AdamW 0.002 — external evaluator agrees.
- **B**: Muon RMS 0.002 — external evaluator disagrees.

The model was correct in JSON format and wrong in compact format on **the same data**. The controller rejected the inconsistent recommendation and produced **no executable or manual review candidate** (`synthetic_review_candidate_only: null`). Three observer/decision-logit anti-forgery tests passed. Exact evidence: `research/results/lfm3_ffn_three_seed_advisor_disagreement_xwing_20261028.json`.

The real-model gradient tests are committed; new seed-bounded LoRA test suite passes on xwing. The model **never changes its own optimizer, grades, loss, budget, weights, source permissions, or routing**. Two failures in earlier LoRA observation settings and this new mixed vote show that the LFM3 observer is **not reliable as an autonomous training controller**. A particular two-prompt success in an earlier one-seed test is not sufficient to relax external voting gates.

## Persistent release blockers

- Tiny fabricated data, repeated development set, no heldout performance or task/generalization evidence.
- No independently authenticated source custody, verified consent for real traces or monotonic signed witness. SHA256 is byte integrity only.
- No durable checkpoint/restart validation for the **actual** LFM3 FFN LoRA. Prior bit-exact AdamW/Muon restart is **in-memory CPU toy** only.
- Hosted `tests` workflow at earlier run `40641029613`: **1,193 passed, 6 skipped, 120 subtests passed**, but **56.62% coverage**, below required **69%**; never lower the threshold.
- SensorGraft parallel journal lock issue #8 reproduces; serial tests cannot prove release-safe locking.

**Next reproducible work:** expand to independently curated synthetic task families, disjoint train/dev/heldout; per-layer optimizer matrix-shape calibration; benchmark real-adapter optimizer state restoration on another host; independent signed observation/custody, fixed policy with rollback and safety budget; then optionally study *genuine* closed-loop synthetic train->observe->review->execute under strict external admission. No production authorization here.
