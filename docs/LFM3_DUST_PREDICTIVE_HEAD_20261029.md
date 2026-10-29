# LFM3 Dust predictive-head forward-only pilot — 2026-10-29

**Status:** draft, isolated, CPU-only research. No release, checkpoint export, online optimizer mutation, training admission relaxation, or automatic promotion.

## Provenance

This is a third *experimental* training route, revived from the prior Dust/K2 work: `auto-finetune` draft PRs #13 (forward-only perturbation), #22 (predictive direction classifier), #23 (real K2 direction witness), and #24 (provenance and split custody). Their earlier confirmation did not show that a trained predictor beat curvature-aware momentum; the K2 pilot's throughput and held-out results did not demonstrate an advantage over backprop. PR #24 also documents shared Unix principal access that undermined supposed independent signing-key custody. None of those negatives is waived.

LFM3 port uses official pinned `LiquidAI/LFM3-1.7B` BF16 base SHA256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` and the text tasks introduced by draft PR #85: 24 procedural training examples, 9 disjoint selection questions and 9 unexamined final-test questions per seed.

## Method

- Attach only to the **two last attention `v_proj` LoRA-B matrices** (2,048 floating-point parameters). LoRA-A and the full pretrained backbone remain frozen.
- Construct a deterministic **8-dimensional perturbation subspace** from a seeded basis. Each step creates four fresh direction coefficients.
- The *causal direction-attention memory* ranks them using weighted similarity to **earlier training-only antithetic probe outcomes**, with bounded memory, recency discount and a strict pre-observation episode boundary. Cold-start order is unchanged. This is a small **external kernel-attention controller**, not a trained or inserted neural attention layer in LFM3.
- Examine only two proposals per step; evaluate positive and negative perturbations on the current **training example**. Apply a bounded update only if it reduces that training loss, subject to a cumulative `0.15` LoRA-B trust radius.
- The random control takes the first two of the same four seeded random proposals. Both receive **exactly 12 antithetic perturbed forward passes** over three steps, independently reset to the same adapter hash and dataset.
- No `.backward()`, autograd gradient step or torch optimizer is invoked. Model parameters have `requires_grad=False`; no gradient tensors should appear. Logs claim zero backward passes; the structural reviewer **cannot cryptographically attest execution**.

This is not a matched adapter-coverage comparison with existing AdamW and factor-wise Muon experiments, which update all 44 Q/V LoRA A/B matrices. No such fair three-way conclusion is drawn.

## Actual 3-seed CPU experiment

Six paired runs of three forward-only training steps each, under explicit `--research-only --allow-backprop-free-lora-probe` flags. Each run required **12 antithetic perturbed forwards**, a total of 72; clean and selection forwards are additional. All six reports passed the structural `lfm3_dust_forward_review`.

| Seed | Selection before | Attended after | Random after | Accepted attended | Accepted random |
| --- | --- | --- | --- | --- | --- |
| 1 | 5/9 | 5/9 | 5/9 | 3/3 | 2/3 |
| 2 | 7/9 | 7/9 | 7/9 | 2/3 | 3/3 |
| 3 | 6/9 | 6/9 | 6/9 | 3/3 | 3/3 |
| **Aggregate** | **18/27** | **18/27** | **18/27** | **8/9** | **8/9** |

Causal attention actually changed selected candidate indices after the first step; the random policy kept the predeclared first two of four random candidates. The selection log-probability margin changes were mixed; no accuracy, generalization, cost-efficiency or statistical superiority was established. No final-test partition was evaluated. A training-loss improvement at a probed example is not evidence of global learning.

Evidence files (all source-only, no personal prompts, adapters or weights) live under `research/evidence/lfm3-dust-forward-20261029/` named `seed{1,2,3}-{attended,random}-3steps.json`.

### Reproduction

With the pinned official local checkpoint and isolated Transformers 5.x compatibility stack, explicit offline CPU environment and a dedicated worktree:

```bash
python -m src.lfm3_dust_predictive_probe \
  --research-only --allow-backprop-free-lora-probe \
  --seed 1 --steps 3 --draws 2 --policy attended \
  --model-path /home/scott/lfm3-research/model-metadata \
  --output /tmp/lfm3-dust-attended-replay.json

python -m src.lfm3_dust_forward_review \
  research/evidence/lfm3-dust-forward-20261029/seed1-attended-3steps.json \
  research/evidence/lfm3-dust-forward-20261029/seed1-random-3steps.json

python -m pytest -o addopts= -q \
  tests/test_lfm3_dust_predictive_probe.py \
  tests/test_lfm3_dust_forward_review.py
```

The experiment runs only with explicit flags, offline environment and CPU-only device controls. Output is exclusive-create; model/adapters are never saved.

## Next research gates

1. **Equal parameter coverage**: compare three methods on exactly the same two frozen LoRA-B tensors and equal paired seeds/data; report forward/backward counts, wall time, quality and memory, not just nominal LR.
2. **Predictive benefit**: preregister an independent episode-disjoint confirmation against random, momentum and curvature-aware direction selection; measure prequential prediction error, improvement per perturbed forward, and cost per successful update. Do not train/choose on final-test examples.
3. **Real LFM3 head**: an activation-conditioned neural prediction head is a separate future experiment. Its own coefficients may be trained analytically (e.g. online ridge) from historical train-only labels **without model backprop**, but this is **not yet implemented**. Current implementation uses only 8-D perturbation direction coordinates and an external attention memory.
4. **Trusted provenance**: independently owned keys/principals, signed per-probe precommit records, source-disjoint data, and honest timing attestation are not yet established. Do not repeat the earlier shared-account signing claim.
5. **Production readiness**: require diverse independently labeled tasks, retention/safety regression checks, calibrated uncertainty, and human-reviewed authorized training controls. No model proposal is allowed to update live weights or bypass the Xwing recovery latch.

The results are evidence of a working mechanism and its costs, **not evidence of recursive self-improvement**.
