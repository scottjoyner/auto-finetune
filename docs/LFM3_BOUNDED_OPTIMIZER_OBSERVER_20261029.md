# LFM3 bounded optimizer observer — research-only v0.1

This changeset is stacked on the real-checkpoint LoRA probe branch, not on live production training. It parses the **verified** synthetic-token AdamW/Muon run receipts. It never calls a model, changes an optimizer, starts GPU work, loads weights, writes a checkpoint, or promotes a model.

## Contract

- Input: `sensorgraft.lfm3.real-lora-synthetic-probe.v1` telemetry; review with the existing strict `lfm3_real_lora_review.review()` first.
- Reject source provenance mismatches, incomplete arms/steps, nonfinite events, suspicious gradient/update spikes, fewer than 3 seeds or 20 steps, and inconsistent learning rates across paired arms.
- Generate a non-executable grid around each already tested optimizer LR: 0.8x, 1.0x and 1.2x where within the existing runner's bounds.
- Optional language-model suggestion is strictly treated as **untrusted data** with exactly four keys: `optimizer`, `learning_rate`, `base_sha256`, `reason`. Only already observed optimizer types and at most a 20% LR deviation are accepted for *advice*.
- All outputs explicitly set `trial_authorized=false` and `weight_promotion_permitted=false`; no production selection logic exists.

## Reproduction

From a checkout containing the full pinned probe evidence:

```bash
python3 -m pytest -q tests/test_lfm3_optimizer_observer.py tests/test_lfm3_real_lora_review.py
python3 -m src.lfm3_optimizer_observer \
  research/evidence/lfm3-real-lora-20261029/lfm3-real-lora-3seed-20step-equal-lr-20261029.json
```

No model weights, GPU, Hugging Face authentication, network connection, or runtime PyTorch dependency are required for the observer. The test suite should run in the same Python environment as the repository's other pure-Python reviews.

## Next experimental stage

1. Acquire licensed, authorized **real task data**, plus completely separate fresh train / selection / final test splits. Hash inputs and retain privacy custody in SensorGraft; do not publish private prompts or tokens.
2. Verify parameter groups for each optimizer. In PyTorch, Muon targets eligible 2D hidden-layer matrices; LoRA factors are still an experimental grouping, and embedding/bias/norm parameters remain on AdamW. Compare AdamW, mixed Muon+AdamW, and justified LORA-specific Muon only when supported by independent evidence.
3. Run paired seeds from identical pinned weights, dataset ordering, and compute budgets. Log token throughput, memory, grad/update norm, optimizer state, train loss, fresh validation loss, retention and task success by domain.
4. Give LFM3 a **read-only telemetry summary** and require a strict, bounded JSON suggestion. Evaluate suggested and control configurations on fresh heldouts in a disposable worker; never allow model text to control the shell, checkpoints or deployment.
5. Add independent model-selection correction (multiple comparisons), safety/retention regression thresholds, rollback checkpoint verification, reviewer sign-off, and signed SensorGraft receipts before any promotion gate is even considered.

## Limitations and interpretation

The underlying run used official LFM3-1.7B weights, a frozen backbone, rank-2 LoRA adapters and synthetic random token IDs. Its held-out loss is not a language-capability measure. The proposed 0.8/1.0/1.2 grid is a research *design*, not a data-backed best learning rate. Timings in this evidence are CPU-only and cannot establish R9700 ROCm throughput. Muon-on-LoRA compatibility is not validated for production. This change does not implement recursive weight self-modification or autonomous tuning; it establishes an auditable observation/feedback boundary for later experiments.

See PyTorch's Muon optimizer documentation and LiquidAI LFM3 documentation for architecture and optimizer constraints.

## Read-only LFM3 self-observation packet (v0.1)

The new `src.lfm3_optimizer_feedback_packet` serializes only verified numeric optimizer summaries, candidate learning rates and pinned digests as a two-message observation prompt for a locally hosted LFM3 model. It does not contact a server, start inference, enqueue training, or write checkpoints; the packet is an **input contract**, not a measured AI-generated improvement.

```bash
python -m src.lfm3_optimizer_feedback_packet \
  research/evidence/lfm3-real-lora-20261029/lfm3-real-lora-3seed-20step-equal-lr-20261029.json
python -m pytest -o addopts= -q tests/test_lfm3_optimizer_feedback_packet.py tests/test_lfm3_optimizer_observer.py
```

Candidate model JSON now requires exactly `optimizer`, `learning_rate`, `base_sha256`, `report_sha256`, `reason`. `report_sha256` is the SHA-256 of the **canonicalized full input telemetry JSON**, protecting a recommendation from accidental stale/replayed telemetry. It is not an authentication signature and does **not** make the source trusted: independent signed SensorGraft provenance is still required. Invalid types, mismatched report identity, nonfinite or out-of-bounds LR, control characters, extra keys and unsupported optimizers fail closed.

At the last read-only x1-370 LM Studio inventory check, a K2-Horizon model was loaded but **no LFM3 model was loaded**. No loaded model was evicted and no LFM3 generation was performed, so there is no claimed autoregressive model-suggested LR result yet.

The 0.8×/1.0×/1.2× grid remains **purely a preregistered trial design**. Independent authorized evaluation, multiple-comparison correction, telemetry signing, and rollback tests must precede any controlled adaptive optimizer setting.

## Optional actual local LFM3 inference (still read-only)

`src.lfm3_optimizer_feedback_client` is an explicitly opted-in, localhost-only bridge to the LM Studio endpoint. It first examines `127.0.0.1:1234/api/v1/models` and only sends the numerical observation packet if exactly one `LFM3-1.7B` model is **already loaded**. It does not load or evict models. This is a **single inference sample**, not a training feedback daemon, not actual autonomous optimizer updates.

```bash
python -m src.lfm3_optimizer_feedback_client \
  research/evidence/lfm3-real-lora-20261029/lfm3-real-lora-3seed-20step-equal-lr-20261029.json \
  --research-only --allow-local-inference
python -m pytest -o addopts= -q tests/test_lfm3_optimizer_feedback_client.py
```

Model text is parsed as strict JSON, then reviewed with exact base and report digests, safe optimizer names, and bounded LR. Invalid responses are never executed. There is no user-configurable endpoint, shell command, optimizer setter, or checkpoint writer. Request timeout is 50 seconds; response cap is 32 KiB; inference cap is 160 tokens. Only pure-Python mocked network contract tests belong in CI. The client itself remains optional and disabled without both CLI opt-in flags.

**Not yet accomplished:** verification against an actually loaded LFM3 model, CPU/ROCm end-to-end model-generated proposals, signed SensorGraft provenance, mixed Muon/AdamW parameter-group optimizer experiments, and independently graded task-quality changes. These must be measured separately. Never use synthetic-token loss to authorize weight promotion.
