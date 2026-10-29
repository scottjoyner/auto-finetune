# Real LFM3 FFN AdamW/Muon native optimizer process-restart experiment

**Date:** October 28, 2026. **Status:** experimental draft; no production deployment or autonomous optimizer changes.

## Objective and actual execution

The existing pinned official BF16 LiquidAI LFM3-1.7B weights (revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`; `model.safetensors` SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`) were loaded **fresh in three separate network-disabled xwing bubblewrap processes per optimizer**.

A rank-4 additive LoRA hook at the real final feed-forward projection `model.layers.23.feed_forward.w2` trained **38,912** adapter parameters via real backpropagation through the frozen model. The **1,728,801,536 pretrained parameters remained untouched**. Only 2 explicitly fabricated training and 2 explicitly fabricated development examples were used; no real user traces, NAS or Neo4j data.

One independent fixed seed (`20261028`) per optimizer, LR `0.002`, exactly three optimizer steps.

- **Baseline:** an uninterrupted three-step run; records development cross-entropy and a canonical SHA-256 over adapter weight tensors, all optimizer parameter groups, step tensors and native state.
- **Save:** a new process performs one training step and writes to a **fresh private** mode-0700 `/tmp/lfm3-durable-*/` directory. The LoRA tensors and native optimizer state (not the 1.7B backbone) are serialized using Torch; output is bounded to 16 MiB, created exclusively with no symlink-following, synced with file `fsync`; a separate strictly typed SHA-256 manifest is also created exclusively and synced, followed by parent-directory `fsync`. An incomplete payload without manifest is refused.
- **Resume:** a third independently launched process reloads the pinned official model, requires the exact manifest/checkpoint/model/optimizer/step identity, verifies bounded ordinary no-hardlink files plus detectable read-time file changes, checks SHA-256, uses Torch `weights_only=True`, verifies the actual two adapter matrices' shapes/dtypes/finiteness, loads both native tensors and optimizer state, then executes the remaining two steps.
- **Independent assessment:** `src/lfm3_ffn_cross_process_gate.py` compares each pair of baseline/restart hashes and dev losses against one another **and** the precommitted three-seed development receipt, rejecting altered source identity or authority flags. This gate is read-only.

All six actual model processes were **rerun using the exact same optimizer/replay source commit `1b1450d4929ebf97f58d20e7bf765027019ca265`**, after resolving Torch `OrderedDict` and zero-dimensional step-tensor hashing edge cases. Only this clean same-source reproduction is retained as the canonical research receipt.

| Native optimizer | Three-step baseline dev CE | Fresh-process restored CE | Full native optimizer+LoRA state SHA-256 |
|---|---:|---:|---|
| AdamW | 11.10757064819336 | 11.10757064819336 | `9f5211372a054e94d6d4cbdd039bff131a7fe2416457b070bb6d17fcfa4c6c0b` |
| Muon `adjust_lr_fn=match_rms_adamw` | 11.945520401000977 | 11.945520401000977 | `5d34ceea34839297864100b0ac31b8c41afd416b46c547eccb7ac382669679d2` |

All baseline/restore hashes and exact floating-point losses matched, for both optimizers. The one-step `manifest_payload_sha256` also matched between save and restore.

**Committed receipts:** `research/results/lfm3_real_ffn_adamw_cross_process_scratch_restart_xwing_20261028.json` and `research/results/lfm3_real_ffn_muon_cross_process_scratch_restart_xwing_20261028.json`.

## Corruption and negative-path tests

`tests/test_lfm3_ffn_durable_process_replay.py` verifies private exact scratch directories, allowed mode/optimizer budget, exclusive writes, bounded reads, symlink and hardlink refusal. `tests/test_lfm3_ffn_durable_checkpoint_faults.py` uses native Torch on xwing and fabricated matrices **without loading an LLM** to verify real roundtrip, changed payload SHA refusal, wrong optimizer, missing manifest, forged production flag and second-write refusal. The four native fault tests passed. `tests/test_lfm3_ffn_cross_process_gate.py` verifies equality and denies forged final state, seed, power-loss claim or model activation.

All checkpoint writes occurred exclusively in new private scratch directories. Checkpoint archive bytes were **not committed to Git or published**, only small metadata receipts. The checkpoint is neither a model-finetune product nor a user-data archive.

## Limitations and risk boundaries

A successful three-process test is **not** a forced crash, power loss, machine reboot, filesystem corruption, multi-node synchronization or cryptographic source-authentication test. A SHA-256 stored beside an attacker-editable payload is a tamper-detection check against accidental alteration, **not an authenticated proof against a same-UID malicious process**. Torch `weights_only=True` reduces unsafe object loading but is not a general sandbox for untrusted files. The strict limits are an isolated research safety layer, not a signed approval capability.

The three-step experiment on two fabricated train and dev examples does **not** establish broad AdamW superiority, a general capability gain, self-awareness or recursive self-improvement. The earlier four-example fictional holdout showed only related task-family transfer. Original LFM3's A/A read-only observation produced a non-executing human-review proposal, but no optimizer update or deployment was authorized.

**Production blockers:** hosted Python coverage remains below enforced 69%; SensorGraft journal parallel-writer issue #8 remains open, with stress failures on both `/tmp` and `/dev/shm`; source-witness issue #9 remains open; independently curated heldout tasks, authenticated human model activation, crash-restart/rollback and fleet policy integration require further review. Keep both PRs draft and production model weights, traces, NAS, Neo4j and routing untouched.
