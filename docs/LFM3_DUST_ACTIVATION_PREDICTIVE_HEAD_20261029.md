# LFM3 activation-conditioned predictive Dust head — 2026-10-29

**Status:** research-only; no production training, checkpoint export, self-authorization or weight promotion.

## Background and what is actually built

This extends the earlier Dust/K2 forward-only and predictive direction work (draft PRs #13 and #22–24), and the LFM3 Dust baseline (draft PR #86). The new `src/lfm3_dust_activation_head.py` attaches a **temporary forward-pre-hook** to the final decoder attention `v_proj` during an **additional prompt-only forward pass**, reduces the projection input into four bounded aggregate scalars and removes the hook immediately. Raw token sequences, private traces and full activations are **not** written to the report. The prompt-only forward has no response target or current candidate perturbation result.

A deterministic **two-head online kernel-attention predictor** combines the four-scalar LFM3 state summary and eight coefficients describing candidate perturbation direction. Both attention heads attend over **previous training-only antithetic forward-loss gains**, never outcomes from the current episode. The head updates its small past-outcome memory, *not neural weights*, via gradient-free observations.

**Naming precision:** This is now an activation-conditioned LFM3 predictor, but **not yet a newly trained or inserted transformer attention layer**. It is an external nonparametric, causal multi-head mechanism. A trainable neural head, learned on truly independent episodes, would be a further research milestone and may use backprop or forward-only coefficient learning if separately designed. This branch does not falsely claim that such a model was trained.

## Real pinned checkpoint CPU experiment

Official pinned LFM3-1.7B, frozen pretrained backbone, and only the two last attention V projection LoRA-B matrices, totaling 2,048 modifiable parameters. Three independent deterministic procedural task seeds; three training steps each; `SIGMA=0.05`, two directions selected out of four candidates and 12 antithetic perturbed forwards per run. Every candidate direction must reduce **training-example** loss under the same `0.15` LoRA-B trust radius before a parameter change is accepted. Selection examples are not used in direction attention; the locked future-test partition remains unevaluated.

| Seed | Random selection after | Direction-only head after | Activation-conditioned head after | Activation accepted |
| --- | --- | --- | --- | --- |
| 1 | 5/9 | 5/9 | 5/9 | 3/3 |
| 2 | 7/9 | 7/9 | 7/9 | 2/3 |
| 3 | 6/9 | 6/9 | 6/9 | 3/3 |

Across the three activation-conditioned runs: 36 antithetic perturbed forwards + **9 additional prompt-only forwards**. Combined with existing direction-only and random evidence, the three-policy comparison totals 108 antithetic perturbed forwards, plus activation-context overhead, clean loss measurements and final selection evaluations. Each activation run reports zero backward calls and no optimizer object, but these are local execution assertions, **not independent or cryptographically attested evidence**.

The activation head changed selected candidates relative to the direction-only baseline on seed 1 step 2 and seed 2 step 3. On seed 3 it made the same choices. **Selection accuracy and final selection margins matched the direction-only head in all three seeds**, despite extra CPU work. There is no evidence of improved learning from conditioning, no statistically meaningful optimizer ranking, and no validated recurring self-improvement. Higher-cost activation extraction should remain opt-in.

Representative isolated CPU run times for each seed: activation head **9.50 / 9.38 / 8.63 seconds**; direction-only **7.56 / 7.49 / 7.83 seconds**; random **7.42 / 7.72 / 8.62 seconds**. Time includes training loop and excludes initial model and tokenizer loading/evaluation outside the loop; do not report these as cross-node or throughput benchmarks. Small three-seed differences are not a robust efficiency estimate.

## Evidence, isolation and review

- Committed unsigned scalar/hash-only reports in `research/evidence/lfm3-dust-activation-20261029/seed{1,2,3}-activation-attended-3steps.json`.
- Independent control evidence in `research/evidence/lfm3-dust-forward-20261029/`. The `lfm3_dust_activation_review` reconstructs the older strict control receipt checks, verifies new per-step SHA-256 context digests, four bounded head disagreement values and exactly one extra prompt forward per step, and checks paired starting adapters, baseline selection scores, data hashes, candidate probe budgets and optimizer scope.
- Strong **negative boundary:** a context digest only allows detecting a changed recorded value; it does *not* prove prompt-only origin, independence, causal timing, no leakage or accurate runtime execution. A separately isolated receipt collector remains required for credible independent research evidence.
- Run `python -m pytest -o addopts= -q tests/test_lfm3_dust_activation_head.py tests/test_lfm3_dust_activation_review.py tests/test_lfm3_dust_predictive_probe.py tests/test_lfm3_dust_forward_review.py` without model weights or internet.

## Next experiment — preregistered, not run

1. Record **prequential (predict-before-measure)** error for each candidate and per head, with fresh independent episode groups. Compare against cheaper direction-only attention, random, momentum and curvature-aware Dust baselines at equal direction probes and **total** forward compute cost.
2. Compute whether activation-conditioned ranking reduces selection regret/achieves better task outcomes. If not, fall back to cheaper direction-only attention. Avoid choosing models by the test split or running any unbounded self-tuning loop.
3. If the state-conditioning effect is reproducible, evaluate an **actual trainable prediction head** with online ridge/regression or forward-only weight updates to its tiny coefficients, while freezing LFM3 and preserving train/selection/final-test boundaries. A trained predictor can still overfit sparse directions and must not control production directly.
4. For fair three-way AdamW/Muon/Dust comparison, restrict *all* algorithms to the same last two LoRA-B tensors, matched time/forward-equivalent budgets and initial weights. Full LoRA AdamW and Muon results from PR #85 are not direct competitors.
5. Verify code and evidence via GitHub CI; keep all draft PRs unmerged and the 69% repo-wide coverage requirement, training-admission stop files, Xwing recovery latch and production configuration intact.

This is a genuine measured step toward feedback-conditioned optimizer research, **not proof of autonomous recursive self-improvement**.
