# Real LFM3 FFN LoRA: development choice, frozen lexical holdout, self-observation and optimizer restart
Date: 2026-10-28
Status: **draft research only; not authorized for deployment, continuous training, or autonomous optimizer changes.**

## 1. Experiment and source custody

Executed on xwing, CPU, network-unshared bubblewrap, using the locally cached official BF16 `LiquidAI/LFM3-1.7B` revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.

Frozen backbone **1,728,801,536 pretrained parameters**. Only a rank-4 additive LoRA hook at `model.layers.23.feed_forward.w2`, **38,912 new trainable parameters**, received real last-layer gradients propagated from official vocabulary logits. No original parameters changed, no adapter weight checkpoint written, no production NAS/Neo4j or real user traces touched.

The existing verified **six-arm development comparison** (three independent seeds, native AdamW LR .002 vs native Muon `adjust_lr_fn="match_rms_adamw"`, 3 train steps each) was independently checked by `src/lfm3_ffn_multiseed_guard.py` **before** admitting the lexical holdout. It had already selected AdamW based on the two-example development set.

A new, frozen, **four-example related fictional codebook holdout** was written separately from the two fictional training and two development prompts, with disjoint exact prompt text and distinct split digests. Its contents were not supplied to the optimizer-selection function. **This is not a independently curated benchmark, not a new task family and not a claim of statistical generalization.** The source/custody and consent remain unattested.

## 2. Six real LFM3 replayed arms and once-only lexical holdout

The research-only evaluator `src/lfm3_ffn_lexical_holdout.py` retrained the six fixed optimizer/seed configurations, verified their resulting development loss agreed with the **previous committed** development receipt before reading that arm's holdout, and performed no optimization after reading heldout values. It also measured the frozen original model's holdout loss as a control.

Baseline frozen model's fictional holdout loss: **9.0082244873**.

| Initialization seed | AdamW heldout CE | RMS-matched Muon heldout CE | Relative AdamW reduction from frozen base |
|---|---:|---:|---:|
| 20261028 | 8.2915935516 | 9.0081920624 | 7.96% |
| 20261029 | 8.3720359802 | 9.0082120895 | 7.06% |
| 20261030 | 8.3807315826 | 9.0082111359 | 6.97% |

The existing precommitted dev losses were reproduced (AdamW 11.107571 / 11.227321 / 11.210147 vs Muon ~11.945520). The new lexical family heldout supports **within-family transfer to four new prompts**, and indicates that AdamW improved this tiny fictional task more than Muon at the specified three-step budget. It cannot establish general LFM3 task performance, model self-improvement, statistically independent trials, or broad optimizer superiority.

The independent `src/lfm3_ffn_holdout_evidence_gate.py` verifies the frozen development receipt and its SHA, exact source split identities, six paired trials, development replay parity, finite measured losses, zero source/promotion authority and independently recomputes all per-seed holdout effects. It only permits **nomination of a new synthetic research experiment for manual review**, not weight updates. Three falsification tests passed.

Exact committed packet: `research/results/lfm3_real_ffn_dev_selected_lexical_holdout_3seed_xwing_20261028.json`.

## 3. Original LFM3 observes its own adapter evidence

`src/lfm3_ffn_holdout_observer.py` loaded the **unchanged official pretrained** LFM3 and gave it two fixed prompt formats with **only externally validated numeric holdout results**. It made two A/B choices by comparing the actual next-token logits, with no sampling or executable tool outputs.

- JSON layout: **A (AdamW)**, logits A=17.5, B=13.25.
- Reordered text layout: **A (AdamW)**, logits A=17.375, B=15.5.
- Independent numeric objective: **A** in all three seeds.

Unlike the previous development-only 2-layout model observation (A/B disagreement and forced veto), this holdout review had **A/A** agreement. The controller emitted only `{"optimizer":"adamw","learning_rate":0.002,"execute":false,"manual_review_required":true}`, which was **not executed**. No optimizer hyperparameter, source, evaluation rule, model state or weights were changed based on the model's advice.

Three fixed-view/logit/authorization regression tests passed. Actual model receipt: `research/results/lfm3_real_ffn_holdout_self_observer_two_layouts_xwing_20261028.json`.

This verifies one narrow observe/compare/propose experiment, **not robust general self-awareness or recursive model improvement**.

## 4. Actual native optimizer state resume, without disk writes

Separate `src/lfm3_ffn_optimizer_replay.py` on the same pinned LFM3 ran:
- Native AdamW at LR .002: uninterrupted three steps vs **one step + in-memory `torch.save` of two adapter matrices and optimizer state + weights-only `torch.load` into freshly instantiated adapters/optimizer + two more steps**.
- Native RMS-matched Muon at LR .002: the same fixed controlled experiment.

For **both** optimizers the final trained adapter tensors, optimizer state dictionaries (including tensor equality), and development loss were **bit-exact** relative to uninterrupted training. The final dev losses were 11.1075706482 (AdamW) and 11.9455204010 (Muon). In-memory packet lengths were **470,413 bytes** and **313,885 bytes**, respectively. The bytes are bounded and SHA-256 validated before weights-only deserialization. Three budget/tamper tests passed and exact model replay was rerun after adding the checkpoint SHA gate.

Original pretrained weights remained frozen, no serialized optimizer state was ever written to disk, and this **does not validate process restart, sudden death, corrupt on-disk recovery, fsync, power loss or multi-node checkpoint consistency**.

Committed first run receipt: `research/results/lfm3_real_ffn_native_adamw_muon_inmemory_optimizer_replay_xwing_20261028.json`.

## 5. SensorGraft: research audit only

Companion SensorGraft draft PR #7 freezes matching holdout and model-vote packets in `research/fixtures/` and independently validates them with `src/lfm3_ffn_holdout_lineage.rs`. The separate categorical graph has **10 nodes, 12 edges** (six measured holdout trials, independent evaluator, two model observations and one explicitly non-executed manual-review proposal). A fresh isolated `/tmp/sg-lfm3-lab-*/` journal was written and replayed at revision 2; three strict Rust tests and strict Clippy passed. The graph does not contain model weights, semantic memory or an authenticated producer witness.

## 6. Release blockers

- Hosted Python test coverage **minimum 69%** is still enforced. Last observed completed test workflow at source `8ccc7829` had **1,223 passed, 6 skipped, 145 subtests**, but overall **59.26%** coverage; no gate weakened. Later optimizer-restart modules require final CI validation.
- SensorGraft **parallel journal lock contention** issue #8 persists despite serial all-target success; no production journal writes.
- Independent source custody/witness, privacy/consent for real telemetry and physically trustworthy monotonic source generation remain unresolved (SensorGraft #9).
- No independent task-family heldout, multi-seed optimizer restart, durable restart, large-scale generalization, ROCm/GPU replay, typed signed human promotion approval or serving-level execution gate.
- LFM3's advisor output can be brittle to prompt formatting; **the external evaluator and manual review retain control**.

Both draft PRs remain unmerged. No fleet services, routes, production model weights, Neo4j or NAS were changed.
