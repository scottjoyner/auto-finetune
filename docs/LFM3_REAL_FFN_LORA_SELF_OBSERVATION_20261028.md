# LFM3 optimizer self-observation: 18 head-LoRA trials and real final-layer backprop

**Date:** October 28, 2026. **Status:** draft PR #61; offline fabricated research, not self-modifying production AI.

## Two distinct experiments (do not conflate)

### A. Three initializations of a frozen-head LM-head LoRA

In `src/lfm3_lora_multiseed_lab.py`, we loaded the **pinned real BF16 LFM3-1.7B** (checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`), extracted fixed final hidden-state features from **2 fictional train and 2 fictional development prompts**, and froze all 1,728,801,536 base parameters. Each independent rank-4 LM-head LoRA contained 520,192 new trainable parameters; no adapter checkpoint was written.

We ran seeds **20261028, 20261029, 20261030** through six exact configurations: AdamW LR=.001/.002, native Muon LR=.001/.002 using `adjust_lr_fn="original"`, and native Muon LR=.001/.002 using `adjust_lr_fn="match_rms_adamw"`. Eight gradient steps each, 18 total runs, same fictional data and update budget.

Median dev cross entropy:

| Optimizer | LR .001 | LR .002 |
| --- | ---: | ---: |
| AdamW | 11.674006 | 10.901610 |
| Muon original | 11.754248 | 11.392284 |
| Muon RMS-matched | 11.589749 | **10.619501** |

The RMS-matched Muon .002 dev advantages relative to AdamW .002 were **0.231715, 0.542781, 0.399210** across the three seeds, and the same Muon arm was the lowest measured dev loss in all three. This reduces one seed-specific explanation, but it is **still one tiny, reused 2-example synthetic dev set, no independent heldout, no real agent tasks, and no pretrained weight updates**. This experiment cannot establish optimizer superiority generally.

Exact research receipt: `research/results/lfm3_real_frozen_head_lora_three_seeds_18_trials_xwing_20261028.json`. The deterministic gate permits only a **new synthetic trial to be nominated for human review**; it never executes a model choice or promotes weights.

### B. Actual full-model forward with a low-rank hook INSIDE LFM3

In `src/lfm3_last_layer_lora_lab.py` we attached a rank-4 forward hook to **`model.layers.23.feed_forward.w2`** in the official model, rather than training on cached final hidden features. We computed cross-entropy from the **real full-model forward pass**, backpropagated to the 38,912 new LoRA parameters (both low-rank matrices after three steps), verified real gradients, and kept all pretrained weights frozen and unchanged.

Two fixed configurations, both LR=.002, same fabricated 2-train/2-dev split and three gradient steps per arm:

| Method | Dev loss before | Dev loss after |
| --- | ---: | ---: |
| AdamW | 11.945557 | **11.107571** |
| Native Muon `match_rms_adamw` | 11.945557 | 11.945520 |

AdamW improved development loss substantially more in this **one-seed three-step real-layer smoke**, whereas Muon was better in the separate **three-seed frozen-output-head** experiment above. This does **not** establish the best optimizer for arbitrary LFM3 matrix shapes, modules or tasks.

The one-step precursor is preserved separately. Exact three-step receipt: `research/results/lfm3_real_last_ffn_lora_3steps_adamw_muon_xwing_20261028.json`.

### C. Pinned LFM3 observing its own final-layer experiment

`src/lfm3_ffn_lora_observer.py` validated the genuine three-step results, computed an **external** dev-loss argmin and minimum effect-size gate, and then reloaded the official **unaltered** LFM3 model read-only to score fixed `A` (AdamW) / `B` (Muon) option tokens in two prespecified prompt formats. LFM3 selected **A / A**, matching external best **A**, and the measured loss gap exceeded the minimum absolute+relative thresholds.

This is a model-driven **proposal record for a possible future synthetic research run only**, with `execute:false` and `requires_manual_review:true`. The observer ran **zero optimizer steps** and had no write access to checkpoint files, policy, data, source trust, fleet routing, evaluator or model weights. Real graph-training adapter weights were deliberately **not saved**; the observer is therefore the pinned base model, *not a newly updated self-training model*. Two correctly ranked fixed examples are insufficient to claim reliable autonomous optimization.

Exact observation: `research/results/lfm3_real_last_ffn_lora_3step_self_observer_xwing_20261028.json`.

## Independent guardrails and next honest release criteria

1. Run **multiple seeds on real internal-layer LoRA** with a broad, independently authored synthetic train/dev suite; reserve a final heldout by source/session and never let the model choose its evaluator. A dev metric decrease alone is not generalization.
2. Compare shape-calibrated Muon against matched AdamW budgets across final FFN, attention, LoRA head and convolution modules **only** after architecture-specific eligibility review.
3. Independently authenticate experiment source generations and privacy/consent. All present SHA-256 hashes assert *byte equality*, **not** custody, signed provenance or production authorization.
4. Validate durable **file-backed** optimizer/checkpoint resume across process crashes, partial writes, corruption, optimizer-group changes and rollback. Existing bit-exact restart is **in-memory CPU toy only**, not a durable fleet controller.
5. Keep the full static evaluation, training-admission, CI coverage 69%, journal concurrency and human review gates unchanged. Never let LFM3 schedule infinite loops, mutate its own reward, or publish weights.
6. Leave PR #61 draft; no production Neo4j, NAS traces, iOS app, fleet models, weights or routing modified.

All experiments used the same verified official BF16 checkpoint inside a separate, network-disabled bubblewrap process with read-only model/research mounts and bounded CPU research memory. Future GPU/ROCm or cross-node performance is unmeasured.
