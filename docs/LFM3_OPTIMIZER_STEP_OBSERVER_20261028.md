# LFM3 optimizer-step observer — research contract

This module is a **read-only analysis tool**, not live optimizer control. No GPU
training, model inference, private trace ingestion, or production promotion occurs.

## Running

Run from the repository root:

```sh
python3 -m src.lfm3_optimizer_telemetry /path/to/optimizer-steps.jsonl
python3 -m pytest -o addopts='' -q tests/test_lfm3_optimizer_telemetry.py
```

The caller/trainer must emit one JSON object per optimizer step using
`sensorgraft.lfm3.optimizer-step.v1`. Required and **only** accepted fields:
`schema`, `run_id` (32 lowercase hex), `step`, `seed`, `base_sha256`,
`dataset_sha256`, `optimizer` (adamw/muon), `learning_rate`, `train_loss`,
`grad_norm`, `update_norm`, and `elapsed_s`.

Do not include prompts, completions, credentials, raw trace text, or paths.
Run IDs and hashes are **claims supplied by the producer**, not independently
attested; a later signed custody layer must verify them before promotion.

## Observer behavior and limits

- Refuses missing/duplicate steps, mixed run IDs, seeds or dataset/model hashes.
- Refuses unknown fields, malformed or nonfinite numeric values, and unsupported
  optimizer names. Caps input at 5 MB or 10,000 events and requires 20 steps.
- Compares the first and last five steps using medians, not individual spikes.
- A flat training-loss curve may generate a **suggestion** to reduce LR by 10%.
- Moderate gradient/loss instability may suggest a 50% LR reduction.
- Severe instability blocks; clear improvement normally produces no proposal.
- Enforces experimental LR bounds, not globally applicable Muon defaults.
- Never applies changes, saves weights, edits service config or grants promotion.

## Next integration gates

1. Wire a disabled-by-default callback into the *current* LFM3 trainer branch;
   verify parameter groups before comparing AdamW and Muon. Muon is primarily
   intended for eligible hidden-layer matrices, not arbitrary bias/LoRA vectors.
2. Record verified optimizer state snapshots and checkpoint hashes; preserve
   model/adapter provenance and compute context (hardware, energy, throughput).
3. Repeat controlled A/B runs with multiple seeds, fixed token budgets, disjoint
   evaluation corpora and optimizer-state rollback.
4. Add independent SensorGraft receipt verification and heldout tests before
   granting any promotion. Training loss alone cannot establish improvement.

References: https://github.com/KellerJordan/Muon ;
https://github.com/meta-pytorch/torchtune (maintenance ended in 2025);
https://github.com/pytorch/ao (quantization roadmap now consolidated).