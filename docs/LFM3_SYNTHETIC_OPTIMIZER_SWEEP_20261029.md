# Synthetic AdamW/Muon A/B experiment and fail-closed sweep review

**Date:** 2026-10-29 (x1-370 fleet orchestrator). **Status:** synthetic surrogate experiment only. NOT an LFM3-1.7B pretrained-model fine-tuning benchmark or evidence of recursive self-improvement.

## Controlled experiment

The src/lfm3_optimizer_ab_harness.py runner executes real CPU PyTorch optimizer steps on a small 16→32→8 MLP, never on pretrained LFM3 or fleet production services. A frozen synthetic teacher creates 128 training examples and 64 held-out examples. Each seed and optimizer pair shares initial weights, data, and batch order, with disjoint heldout rows verified. Model/data/final-state SHA-256 fingerprints and unique execution IDs are recorded.

Three seeds (1, 2, 3), 50 steps and batch size 16 were used. AdamW controls all parameters at learning rate 0.0001. In the hybrid Muon arm, Muon controls only the two 2D weight matrices, while AdamW handles the two 1D biases at a **fixed learning rate of 0.0001 across all experiments**. Weight decay was zero.

Environment: x1-370, PyTorch 2.9.1+rocm6.4 in existing venv, CPU execution with 2 threads. This is not controlled speed benchmarking.

## Corrected synthetic held-out results (mean over 3 seeds)

| Muon matrix learning rate | Mean Muon held-out MSE | Mean AdamW held-out MSE |
|---|---:|---:|
| 0.0001 | 0.376473 | 0.359088 |
| 0.0005 | 0.352010 | 0.359088 |
| 0.0010 | 0.323132 | 0.359088 |
| 0.0020 | 0.270630 | 0.359088 |

The reviewer verifies all 24 independent optimizer executions, matched baseline results, unchanged bias learning rates, source fingerprints, step telemetry and heldout evaluation. It outputs review_required with no promotion. Its best observed Muon setting is 0.002, already the configured upper bound, so there is no proposal to increase the rate. Since these holdouts were used to select a rate, **new independent heldout data are required** before accepting any candidate.

Earlier exploratory runs saved under /home/scott/lfm3-research/adapter-candidates were superseded because the initial Muon arm also raised the AdamW bias learning rate; those results should not be cited as a controlled optimizer comparison. Corrected results here keep the bias learning rate fixed.

## Reproduction (synthetic CPU only)

~~~sh
mkdir -p /tmp/lfm3-synth-lab
for muon_lr in 0.0001 0.0005 0.001 0.002; do
  python3 -m src.lfm3_optimizer_ab_harness --synthetic-only \
    --steps 50 --seeds 1 2 3 --adamw-lr 0.0001 \
    --muon-lr "$muon_lr" \
    --output "/tmp/lfm3-synth-lab/ab-$muon_lr.json"
done
python3 -m src.lfm3_optimizer_sweep_review /tmp/lfm3-synth-lab/ab-*.json
~~~

The harness uses exclusive file creation; it cannot overwrite existing reports. The reviewer blocks malformed, incomplete, mixed-seed, drifted, non-disjoint, duplicate, out-of-bounds or promotion-marked trials. Source files and full synthetic scalar step traces are in research/evidence/lfm3-synthetic-optimizer-ab-20261029/. No private prompts, real datasets, adapter weights or credentials are included.

## Outstanding real-model requirements

Before enabling autonomous LFM3 hyperparameter adaptation: attach an opt-in telemetry hook to the pinned official LFM3 trainer, validate optimizer group compatibility, run multi-seed real LoRA experiments on genuinely new held-out data, compare stability and quality with checkpoint-level rollback, and resolve the CI admission/coverage failures. None of those release gates is cleared by this synthetic experiment.