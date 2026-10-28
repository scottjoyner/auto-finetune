# Real LFM3 FFN adapter: native Muon scaling comparison and activation boundary

**October 28, 2026 | Research only | auto-finetune draft PR #61**

## Actual six-training-run result

The pinned official BF16 `LiquidAI/LFM3-1.7B` checkpoint revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, model-weight SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`, ran in network-disabled xwing bubblewrap with read-only model/source mounts.

Experiment: a **genuine graph-backpropagated final-FFN rank-4 adapter** at `model.layers.23.feed_forward.w2` (38,912 new trainable LoRA parameters). Original 1,728,801,536 LFM3 parameters were frozen; both LoRA matrices received gradients and changed; no adapter or pretrained weights were saved. Three independent initialization seeds (20261028, 20261029, 20261030); each ran three gradient steps with native PyTorch Muon and LR 0.002. Only `adjust_lr_fn` differed: `original` vs `match_rms_adamw`. Two deliberately fabricated training and two fabricated development examples.

| Seed | Muon original dev loss | Muon RMS-matched dev loss | Absolute loss gap |
|---|---:|---:|---:|
| 20261028 | 11.9455242157 | 11.9455204010 | 0.0000038147 |
| 20261029 | 11.9455242157 | 11.9455204010 | 0.0000038147 |
| 20261030 | 11.9455242157 | 11.9455204010 | 0.0000038147 |

Both barely changed from the shared baseline **11.9455566406**. These tiny, BF16-sensitive differences do **not** show meaningful optimizer improvement. Compare the **separate** prior three-seed AdamW-vs-Muon internal-layer experiment, where AdamW dev losses 11.107571/11.227321/11.210147 were materially lower than RMS Muon's ~11.945520. Still, one synthetic task family, no independent heldout and no generalization.

Exact evidence: `research/results/lfm3_real_ffn_muon_original_vs_rms_three_seeds_xwing_20261028.json`.

The independent evaluator `src/lfm3_muon_scaling_evidence_gate.py` requires the exact pinned source identity, six paired arms, optimizer label and three-step budget, rank/parameter count, finite gradients and losses, original-weight integrity, and a **minimum 0.05 absolute and 0.5% relative dev gap for each seed** before merely allowing a *new research proposal* to be reviewed. The measured gap is ~0.0000038, so the gate refuses even research nomination, and **never changes optimizer settings or model weights**. Three focused anti-forgery tests passed on the exact GitHub head.

## Direct registry promotion safety

The existing `src/auto_promote.py` default manual review was not enough because direct `ModelRegistry.promote()` and `complete_ab_test()` could still activate models. The draft implementation now requires **explicit `operator_confirmed=True`** at either central method, and the CLI requires an explicit `--confirm-activation` argument for those two actions. With no confirmation, even a completed A/B sample comparison has no activation side effect.

`ModelRegistry.start_ab_test()` now retains the actively serving baseline status, while marking only nonactive models as testing. A/B metrics reject nonfinite/invalid scores, bool/string spoofing, reserved `_count` metrics, and unauthorized versions. Completing an A/B test additionally requires nonzero per-arm numeric score counts.

`AutoPromoter` default remains human-review-only; its **explicit legacy opt-out** passes the registry confirmation flag. These keywords are **not authenticated credentials** and do not prove the caller is a person or has approval authority. A real release still needs a separately authenticated signed/verified capability enforced at the model-serving boundary. Do not claim the current boolean is a production security mechanism.

**17 focused tests** across the registry, auto-promotion, feedback and alerting passed on xwing, plus three standalone numerical Muon scaling tests; all use mock or fabricated data and temporary files. Exact GitHub CI admission passed for previous commits; newer final-head CI must be checked before release claims.

## Remaining blockers

- Full hosted Python **69% coverage** remains enforced. Last observed completed run before additional tests: 58.95%, passed 1,208 tests; only coverage failed.
- SensorGraft journal **parallel lock conflicts** (issue #8) remain unresolved; serial green does not clear concurrency.
- No independent heldout tasks, authenticated source provenance, real LFM3 adapter durable restart test, multi-layer architecture-specific optimizer calibration, or tested production rollback.
- Neither draft PR may be merged for production or initiate continual optimizer updates.

Next experiments should vary parameter shape, separate real FFN/attention/LM-head adapter groups, add frozen holdout families once, and compare native AdamW and Muon under matched update norms. LFM3 may **observe** metrics, but the external fixed evaluation/admission controller decides whether new research can be proposed.
