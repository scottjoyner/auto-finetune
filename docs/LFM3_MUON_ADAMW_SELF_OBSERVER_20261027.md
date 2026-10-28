# LFM3 observing AdamW/Muon fine-tuning telemetry: guarded research handoff

**2026-10-27, auto-finetune draft PR #61. Research only.** This is a first observer/optimizer experimental loop, **not recursive autonomous fine-tuning of LFM3**. All training was on an independently initialized small synthetic network. LFM3's pinned BF16 weights were loaded **read-only for observation and parameter inventory**, never trained or modified. Production SensorGraft traces/Neo4j, fleet routes, and weight checkpoints were untouched.

## Current verified pipeline

1. `src/lfm3_optimizer_toy.py` initializes a fabricated small two-layer network with exactly the same seed and train/dev tensors for two isolated arms: native `torch.optim.AdamW` on all trainables, and native `torch.optim.Muon` on the explicitly reviewed 2-D `hidden.weight` with AdamW on all other parameters. Each arm creates fresh optimizer state and runs 24 real gradient steps on CPU, with no saved model weights.
2. `src/lfm3_optimizer_guard.py` validates strict source-scoped numeric observations, bounds learning rates, decay, momentum, Newton–Schulz steps, total experiment cycles, and train/dev split identifiers. Heldout scores/answers do **not** reach the observer. `compare_dev_only` requires improvement ≥2%, no nonfinite updates, bounded gradient/latency/memory regression and identical seed, budget and dataset identity. Passing this check can only yield a candidate for manual review, **never automatic promotion**.
3. `src/lfm3_optimizer_advisor_lab.py` fed LFM3-1.7B the numeric toy AdamW development observation. The generated free-form answer was **rejected** by strict JSON admission. It executed no action.
4. `src/lfm3_optimizer_choice_lab.py` instead scored only **six fixed single-token choice labels** using the pinned BF16 LFM3 next-token logits. No free-form model code or JSON is interpreted. The deterministic controller translates the selected label to a bounded `new-synthetic-experiment-only` proposal. LFM3 selected **A: repeat AdamW at unchanged LR 0.001**. This is a **no-change** outcome, *not optimizer progress*; no additional training was triggered by that choice.
5. `src/lfm3_optimizer_partition.py` and `src/lfm3_optimizer_inventory_lab.py` inventory the actual pinned LFM3 architecture with no gradients or optimizer state. **Muon candidacy is restricted to feed-forward W1/W2/W3 and attention Q/K/V/O dense matrices.** Convolutions, 2-D conv projections, head-wise gate projections, norms, embeddings, output heads and LoRA adapters default to AdamW/unreviewed.

## Actual toy training result

Two **fabricated** arms, same 24 steps, seed 71, small CPU float32 network:

| Metric | AdamW all trainables | Muon hidden + AdamW rest |
|---|---:|---:|
| First train loss | 0.835321 | 0.835321 |
| Final train loss | **0.624481** | 0.699030 |
| Development loss | **0.670682** | 0.739870 |
| Nonfinite updates | 0 | 0 |

Muon hybrid development loss was ~10.3% *higher* than AdamW. The candidate failed the deterministic development-quality gate. Initial timing data are **not comparative benchmarks** because the first arm included library/runtime warm-up and the memory readings are process-level high-water marks; compare performance only with randomized/warmed replicated runs.

**Saved receipts (actual tests):**

- `research/results/lfm3_toy_adamw_muon_pair_20261027.json`
- `research/results/lfm3_optimizer_freeform_advisor_rejected_20261027.json`
- `research/results/lfm3_optimizer_fixed_menu_choice_20261027.json`

## Official BF16 checkpoint parameter inventory

Official model revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`; full BF16 checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.

- **286 tensors; 1,728,801,536 parameters** total.
- **116 candidate Muon 2-D hidden matrices; 1,247,805,440 parameters** (72 feed-forward matrices/1,132,462,080 parameters and 44 attention matrices/115,343,360 parameters).
- **480,996,096 parameters remain AdamW/unreviewed**: 39 convolution/convolution-projection tensors (218,290,176 params), two embedding/output tensors (262,146,048 params), 11 gated-attention projections (360,448 params), and 118 norm/bias tensors (199,424 params).
- `research/results/lfm3_bf16_optimizer_groups_xwing_20261027.json` pins full name/shape inventory SHA and category counts. **Candidate group classification does not mean safe or feasible to allocate full optimizer states on xwing.** No real LFM3 parameters were updated, no adapter trainables were enumerated, and no Muon optimizer was allocated for the full model.

## Safety invariants and compatibility

- LFM3 text outputs and token logits are *untrusted suggestions*. There is no self-edit of code, reward functions, training data, test sets, release gates or a running optimizer. Any new optimizer uses a fresh run with new optimizer state; otherwise AdamW/Muon state cannot be safely swapped.
- No training on production trace archives, no training/evaluation session overlap, no heldout feedback during tuning, no external endpoint actions, no autonomous repetition, and no weight merging or fleet rollout.
- `torch.optim.Muon` is not available in all Torch installations: the hosted CI Torch lacks it. Tests now explicitly verify refusal on unsupported runtimes rather than silently substituting AdamW. Xwing `torch==2.14.1+cu130` ran the actual Muon and AdamW steps on CPU.
- All model loading/observation uses a **separate unprivileged network-disabled bubblewrap sandbox**, read-only official checkpoint and files, pinned Transformers 5.21.0. Python flags alone are not isolation.
- Muon uses matrix-momentum orthogonalization. PyTorch documents using it for 2-D hidden weights with AdamW for other parameters: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html. Its shape-based fallback is insufficient for LFM3's dense convolution projections and gate projections; manual review matters.
- Liquid AI's Oct 27 release describes LFM3's tri-hybrid mix and specialization intent: https://www.liquid.ai/blog/lfm3. It does not establish model-driven optimizer scheduling as a supported native inference API.

## Next engineering milestones (not yet completed)

1. Run **a properly isolated, reviewed, synthetic LFM3 LoRA/adapter fine-tune** with bounded VRAM, private scratch, verified training admission and exact train/dev/heldout session custody; retain AdamW as baseline. Muon on low-rank LoRA is not a default.
2. Have SensorGraft journal each optimizer trial as an untrusted research lineage node with source fingerprint, seed, split, full config, training-state checkpoint and comparison edges. The companion SensorGraft PR #7 now contains a *synthetic-only* 2-node/1-edge graph implementation; still no independent source-generation witness.
3. Run a controlled optimizer grid with warm-up, matched update budgets, several random seeds, gradient anomaly tests, memory/power logging, heldout only at final review, and strict rollback. No one synthetic win counts as improvement.
4. Introduce a separately authenticated monotonic experiment witness and deterministic promotion gate. Never let LFM3 or an auto-generated reward directly decide its own deployment.
5. Fix existing CI coverage gap to ≥69% **without relaxing the threshold**, and investigate SensorGraft parallel journal lock contention (issue #8).

**Current status: both PRs remain drafts and unmerged.**
