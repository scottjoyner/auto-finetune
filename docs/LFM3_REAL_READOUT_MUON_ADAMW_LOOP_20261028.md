# LFM3 optimizer observer iteration — frozen readout and Muon, 2026-10-28

**Scope:** auto-finetune draft PR #61, isolated xwing CPU/bubblewrap research. All prompts/data fabricated; no production NAS traces, real user data, trained LFM3 weights, model checkpoint writes, autonomous deploy or optimizer-state mutation.

## New implemented experiments

- `src/lfm3_optimizer_toy.py` now takes a bounded, independently identified synthetic seed. New seeds carry distinct train/dev fingerprints; original seed 71 matches its frozen earlier receipt.
- `src/lfm3_optimizer_multiseed_toy.py` runs 3 seeds × 6 optimizer/LR arms (18 genuine native PyTorch toy-network training runs), requiring distinct train/dev identities and disallowing nonfinite results. Its deterministic policy vetoes a would-be candidate that regresses against baseline by >2% on any seed. Evidence: initial xwing stdout. These remain **toy** results, not LFM3.
- `src/lfm3_frozen_readout_lab.py` loads the official **real BF16 LFM3-1.7B** checkpoint in a network-disabled, read-only sandbox; fully freezes the 1,728,801,536 backbone parameters, extracts only the last hidden state from **two fabricated train and two separate fabricated dev prompts**, and trains freshly initialized, rank-4, 16,384-parameter final-hidden residual adapters with real gradients. The original full-vocabulary LM head is frozen and used to compute cross-entropy. Only residual parameters enter the optimizer. Backbone in-process tensor version counters are unchanged; no model or adapter weights are saved.
- Default AdamW LR 0.001 and 0.002 each run six steps. A separately explicit `--toy-muon-readout-explicit-exception` permits research **Muon on the two new, highly rectangular residual matrices only**, not LoRA or pretrained LFM3 matrices. It fails if native torch.optim.Muon is unavailable; hosted-CI compatibility is tested. Muon uses `adjust_lr_fn="original"`, NS steps 5.
- `src/lfm3_frozen_readout_observer.py` and `src/lfm3_readout_grid_observer.py` load the pinned BF16 LFM3 model **read-only a second time**, score only fixed A/B or A/B/C/D token options, and use an independent objective evaluator to admit/reject the recommendation as **data**. No free-form output or dynamic reconfiguration is executed.

## Actual frozen-backbone readout outcomes

| Research optimizer | LR | Train loss after six steps | Dev loss after six steps |
|---|---:|---:|---:|
| AdamW | 0.001 | 2.292132 | 5.105943 |
| **AdamW** | **0.002** | **0.529552** | **0.883326** |
| Muon (readout residual only) | 0.001 | 9.742909 | 11.945541 |
| Muon (readout residual only) | 0.002 | 9.491808 | 11.845072 |

All start from train loss 9.836003 and dev loss 11.945557. Muon barely changed the development loss at these calibration settings. This **does not disprove** Muon as a larger-matrix optimizer; the readout residual is rank-4 and exceptionally rectangular, with uncalibrated optimizer scaling. Likewise, improvement in this two-example fabricated dev set **does not establish generalization**.

Saved exact tested result receipts:
- `research/results/lfm3_real_frozen_readout_adamw_pair_xwing_20261028.json`
- `research/results/lfm3_frozen_readout_self_observation_xwing_20261028.json` (two-choice model selected B, agreeing with lower dev loss)
- `research/results/lfm3_real_frozen_readout_muon_adamw_xwing_20261028.json`
- `research/results/lfm3_frozen_readout_four_choice_advisor_refused_20261028.json` (four-choice model selected A, external dev argmin B: **refused**)

In the **earlier toy-network** six-arm setting, LFM3 selected A when C scored best; a follow-on three-seed toy run found AdamW 0.002 lowest in all three seeds. Model choice depends on the displayed task and number of options, and **cannot replace deterministic optimizer admission**.

## Reproduction and boundaries

Use a genuinely unprivileged, network-disabled bubblewrap sandbox with read-only `src`, read-only Hugging Face official pinned snapshot, no `/nas`, no production DB credentials, no writable model paths. Flags inside the Python script are not a substitute for the sandbox.

```bash
# Run only INSIDE the externally enforced offline, read-only sandbox.
python3 -m src.lfm3_frozen_readout_lab \
  --offline-only --ack-sandbox-research \
  --snapshot /approved/local/hf/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd

# Explicitly experimental research on the *new residual*, not base weights.
python3 -m src.lfm3_frozen_readout_lab \
  --offline-only --ack-sandbox-research \
  --toy-muon-readout-explicit-exception \
  --snapshot /approved/local/hf/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd
```

The path is illustrative and must be the real pinned snapshot, verified by `lfm3_checkpoint_attestor.py`. Official checkpoint SHA-256: `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. Exact host runtime: `torch 2.14.1+cu130` with CPU execution and isolated `transformers==5.21.0`.

**Limitations:** 2 synthetic train + 2 synthetic dev examples; one adapter initialization; 6 steps; no actual agent tasks, tool-use evaluation, heldout assessment, independent source custody, generalization claim, statistically meaningful optimizer comparison, multi-node fleet performance or durable optimizer-state recovery.

**Next gates:** independent custody witness; user-reviewed training tasks and disjoint heldout sessions; calibrated Muon matrix shape policy; multi-seed adapter-scale repetitions; proper full sequence/LoRA SFT admission; independent quality/energy/memory/latency regression; hardened optimizer-state resume; new trace lineage authorized only after source provenance. The model must never rewrite its own grader, source permissions, or production promotion logic.

CI: keep the existing **69% coverage requirement**; hosted Torch without native Muon must explicitly refuse the Muon experiment rather than silently substituting AdamW. No merge or deploy.
