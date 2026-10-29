# LFM3 self-observed optimizer experiments: shadow loop and trial-review gate

**Status:** draft research, not production. Source: pinned official LFM3-1.7B, synthetic random-token LoRA AdamW/Muon experiments with 3 paired seeds and 20 real update steps per arm. No demonstrated task-quality benefit or recursive self-improvement.

## Architecture

1. `lfm3_real_lora_review` rejects corrupt, non-paired or non-reproducible source telemetry.
2. `lfm3_optimizer_observer` generates a digest-bound, numerically bounded experiment grid; model feedback must match the exact canonicalized report SHA-256.
3. `lfm3_optimizer_feedback_packet` prepares a minimal numeric-only observation for LFM3. `lfm3_optimizer_feedback_client` optionally queries an **already-loaded** LFM3-1.7B on loopback, and returns only a JSON review verdict.
4. `lfm3_optimizer_shadow_replay` deterministically replays a conservative policy from prior training-step losses, gradient norms and update norms. It decides every five steps, moves LR by at most 0.90x/1.05x per decision, enforces a 20% envelope and three-change cap. It does **not** affect the optimizer: `weight_updates_performed=0`.
5. `lfm3_optimizer_trial_review` separately checks a key-authenticated, paired **external evaluation receipt** with at least three seeds, disjoint split identities and task, retention, safety and throughput observations. It only raises provisional regression flags and requires manual review regardless of metrics.

## How to reproduce

```sh
python -m pytest -o addopts= -q \
  tests/test_lfm3_real_lora_review.py \
  tests/test_lfm3_optimizer_observer.py \
  tests/test_lfm3_optimizer_feedback_packet.py \
  tests/test_lfm3_optimizer_feedback_client.py \
  tests/test_lfm3_optimizer_shadow_replay.py \
  tests/test_lfm3_optimizer_trial_review.py

python -m src.lfm3_optimizer_shadow_replay \
  research/evidence/lfm3-real-lora-20261029/lfm3-real-lora-3seed-20step-equal-lr-20261029.json
```

These operations are pure CPU, do not require model weights or network, and do not touch running training. The separate optional LFM3 inference path is documented in `LFM3_BOUNDED_OPTIMIZER_OBSERVER_20261029.md`.

## Evaluation receipt trust model

An independent test harness must produce a receipt in `sensorgraft.lfm3.paired-optimizer-evaluation.v1` with the source telemetry digest, base checkpoint digest, distinct new train/selection split digests, three paired seeds, separate run identities and per-seed task/retention/safety/throughput metrics. The harness computes a detached HMAC-SHA256 over the canonical JSON payload (excluding `hmac_sha256`) using a trusted evaluator-only key held outside the model or repo. Verification requires a secret of at least 32 bytes.

HMAC confirms a holder of that secret signed the receipt; it **does not** establish evaluator independence, licensing, data privacy, that the split contents are truly disjoint, correctness of claimed human labels, or statistical validity. All of these remain separate review gates. The result always explicitly says `trial_authorized=false` and `weight_promotion_permitted=false`. Never copy the HMAC key into a prompt, trace, GitHub Action or model environment. `LFM3_EVAL_HMAC_KEY` is only a development integration interface and must not be populated in shared CI.

## Scientific and engineering limitations

The synthetic random-token experiment's held-out loss cannot justify AdamW/Muon optimizer selection on real tasks. The moving-window thresholds are illustrative controller rules, not tuned or empirically validated hyperparameters. The trial-review thresholds of 2pp task/retention regression, any new safety regressions, and 15% throughput regression are *provisional screening heuristics*, not proof of generalization or statistical significance. No multiple-testing correction, task stratification, calibration, or prospective human adjudication exists yet.

For a real experiment: freeze source/model/tokenizer hashes; run isolated randomized AdamW, mixed-Muon/AdamW and justified Muon LoRA grouping only after architecture review; keep separate train, model-selection and locked final holdout splits; log actual gradient/update trajectories, effective LR, optimizer state, memory, time and per-domain quality; include negative/sham controls and rollback proof. Only after independent replay and review should a separate, explicitly authorized controller ever change a live training optimizer.

**Protected state:** no production merge, autonomous gradient update, model checkpoint write, stop-file removal, recovery-latch release, resource eviction, or deployment is part of this PR.

## October 2026 optimizer literature audit and experiment matrix

**Critical issue:** applying ordinary `torch.optim.Muon` independently to LoRA A/B factor matrices is *not mathematically equivalent* to Muon applied to the effective full-rank weight update. Our initial Muon arm is therefore an experimental factor-wise reference, not a validated low-rank Muon solution.

- **PyTorch Muon:** specifically designed for two-dimensional hidden-layer parameters. Bias, norm and embedding parameters belong on AdamW or another compatible optimizer. Review `adjust_lr_fn` semantics and count the actual per-parameter effective learning rates before comparisons: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- **ICML 2026, Qu et al., Can Muon Fine-tune Adam-Pretrained Models?:** warns that switching Adam-pretrained weights naively to Muon can disrupt learned features; constraining update strength and LoRA can mitigate optimizer mismatch. Code: https://github.com/XingyuQu/muon-finetune ; paper: https://proceedings.mlr.press/v306/qu26e.html
- **LoRA-Muon, 2026:** https://arxiv.org/abs/2606.12921 — an explicit low-rank spectral update, more appropriate as a genuine Muon-on-LoRA comparison than independent factor-wise Muon.
- **sMuon, 2026:** https://arxiv.org/abs/2608.14492 — an approximate low-rank orthogonal update to inspect for reproducible, compatible reference implementations.
- **PoLoRA, July 2026:** https://parshakova.github.io/polora/ — a product-aware, preconditioned orthogonalized LoRA optimizer; reproductions must independently confirm any reported compute/quality gains on LFM3.
- **Riemannion, ICLR 2026:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/8505a7e1da40320bf5692d7edec5aa45-Abstract-Conference.html — treats low-rank adapters directly on a fixed-rank matrix manifold.

**Proposed prospective experiment arms**, after compatible implementation/licensing review: A0 AdamW LoRA rank-2 control; A1 original factor-wise Muon LoRA as a diagnostic only; A2 mixed Muon (eligible 2D full hidden weights) + AdamW (other parameters) as a separately scoped *non-LoRA* experiment; A3 proper low-rank LoRA-Muon; A4 sMuon; A5 PoLoRA. Re-run at multiple ranks and matched compute/gradient-update budgets, with a locked final test and strict aggregate/individual safety and retention thresholds. Do not compare optimizer learning rates as equivalent across update rules without validating effective update scale and per-step cost.

This is a research shortlist only: no third-party optimizer was installed, executed, licensed for redistribution, or integrated into the training harness here.
