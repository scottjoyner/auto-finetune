# Pinned LFM3-1.7B three-way equal-parameter optimizer experiment — Oct 29, 2026

**Status:** research-only, three seeds, three steps per arm. No quality improvement established, no optimizer selected or promoted, no model checkpoint or adapter saved. The existing 69% full-repository coverage threshold and all protected Xwing admission/recovery controls remain unchanged.

## Question

Can the prior Dust/K2 backprop-free predictive work (PRs #13, #22–24), including the LFM3 activation-conditioned and online calibrated direction-attention controller in PRs #86–88, be a **third parameter-update method** beside AdamW and experimental factor-wise Muon? **Yes, as a bounded experimental mechanism.** Earlier results compared a 2,048-parameter Dust subset with a 146,432-parameter AdamW/Muon LoRA training region. This experiment eliminates that misleading mismatch. It does **not** solve the next scientific requirement of matching compute expenditure, independent evaluation or downstream retention.

## Exact experiment

- Pinned official pretrained BF16 LiquidAI/LFM3-1.7B base, SHA256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`, offline x1-370 CPU-only, no network or production inference.
- Every arm starts from identical per-seed initial last-two-attention `v_proj` LoRA-B matrices, exactly **2,048 trainable/changeable scalars**. The LFM3 backbone, all LoRA-A factors and all other LoRA-B parameters are frozen. Training examples are deterministic public procedural parity, greater-than, addition, with 24 train items, nine *selection* examples and nine untouched final-test items per seed; the final test is **not evaluated**.
- AdamW LR `1e-4`, experimental factor-wise Muon LR `1.6e-4`, each one backward/optimizer step per training example for three steps, gradient max-norm clipped at 1, cumulative adapter trust radius <=0.15.
- Dust random / direction-only / activation-conditioned / online-calibrated heads on the **same starting adapter subset**: three steps, two of four candidate directions per step, four positive/negative perturbed forwards per step and the same 0.15 trust radius. Activation-conditioned policies additionally read prompt-only final-attention aggregate features with one extra full-model forward each step. The online-calibration gate changes small **non-transformer** two-head weights using prior train-only forward-loss prediction errors without backpropagation.
- Selection metric: forced-choice correct-versus-incorrect answer log-probability, **not free-generation quality**. Results below are observational and not a prospective independent final-test claim.

## Observed results: identical parameter scope, unequal compute

| Seed | Before | AdamW | Experimental Muon | Dust random | Dust directional | Dust activation | Dust calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5/9 | 5/9 | 5/9 | 5/9 | 5/9 | 5/9 | 5/9 |
| 2 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 |
| 3 | 6/9 | 6/9 | 6/9 | 6/9 | 6/9 | 6/9 | 6/9 |

Across the new six gradient arms: 18 actual backward/update steps; no held-out score gains. Those records are in `research/evidence/lfm3-dust-matched-three-way-20261029/seed{1,2,3}-{adamw,muon}-matched-b3.json` and pass `src/lfm3_dust_matched_gradient_review.py`. The three-seed paired reviewer verifies the **exact same initial adapter digest**, dataset split hashes, parameter count, source model, step count, trust radius, and explicitly refuses any claim of matched compute, model superiority, or release authority.

Measured training-loop elapsed seconds, excluding initial model load and selection evaluation:

| Optimizer or forward-only policy | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| AdamW, B-only | 4.77 | 6.36 | 4.86 | 5.33 |
| Experimental Muon, B-only | 5.17 | 6.04 | 4.74 | 5.32 |
| Dust random, B-only | 7.42 | 7.72 | 8.62 | 7.92 |
| Dust direction-only attention, B-only | 7.56 | 7.49 | 7.83 | 7.63 |
| Dust activation-conditioned attention, B-only | 9.50 | 9.38 | 8.63 | 9.17 |

Times were observed in separate sequential experiments on one host, not interleaved, and do not establish normalized throughput. AdamW/Muon use gradient computation, while Dust expends at least 12 perturbed forwards per run plus clean/acceptance checks; activation adds at least three extra full forwards per run. They cannot be treated as an equal-FLOP or equal-walltime contest.

## Engineering implications

1. **A trainable predictive-control head already exists in research**, but its learning is limited to bounded online reweighting of two external attention predictors. It ranks which Dust directions get probed and learns from past scalar outcomes. It does not yet implement a new transformer attention layer within LFM3, nor does it directly and safely rewrite LFM3 weights.
2. **Dust is genuinely backprop-free in the isolated implementation**, where model gradients remain absent and bounded B-only changes follow measured forward losses. Source JSON asserts the lack of backward calls; independently signed execution custody still needs to be built before trusting deployment claims.
3. **No method has demonstrated a three-step task-accuracy improvement**. The 9-item selection split is extremely small and repeatedly inspected in research. Do not label it a reliable statistical benchmark or use it as a production decision criterion.
4. **Better predictive tuning requires independent labels**: freeze genuine task families, prospectively stratify train/validation/test, compare gain-per-forward, prequential forecast Brier/MSE, calibration, retention/safety changes and walltime. Add random, momentum and curvature-aware Dust controls; the earlier Dust learned classifier did *not* beat curvature-aware momentum on independent synthetic confirmation.
5. **Adaptive optimizer control is still advisory**: the LFM3 observer/reviewer in PRs #81/#84 can suggest bounded LR experiments, but no result authorizes automatically choosing AdamW/Muon/Dust, changing running training optimizers, saving hot-swap neurons or touching production services. Keep training-admission locks and human independent promotion.

## Reproduce validation

```sh
python -m pytest -o addopts= -q \
  tests/test_lfm3_dust_matched_gradient_probe.py \
  tests/test_lfm3_dust_matched_gradient_review.py
```

All report verification and contract tests run offline on CPU **without loading model weights**. To reproduce actual new pretrained experiments, use the explicit `--research-only --allow-matched-lora-b-gradient-control` flags, add `--allow-experimental-muon-lora` for Muon, and execute only on the isolated pinned research worktree with GPU-disabling and offline environment variables. Output is exclusive-create. Full empirical reproducibility also requires a fresh independent execution and custody review.

**Next preregistration:** equal walltime and matched *total* model forward/backward equivalents with honest accounting; >3 seeds and a genuinely independent real-task quality benchmark; test a trained low-cost head versus direction-only, random, momentum and curvature-aware baselines. Increase experiment duration only in a separately reviewed resource-bounded worktree after independent quality evidence exists.
