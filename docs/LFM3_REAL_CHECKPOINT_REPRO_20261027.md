# LFM3 full-checkpoint research smoke — reproducibility and boundaries

**Date:** 2026-10-27. **State:** successful CPU BF16 full-checkpoint attention hook on xwing; draft PR, not approved for production.

This experiment upgrades the earlier mocked-attention shape tests to real model weights and actual `Lfm3ForCausalLM` forward calls. It is NOT evidence that external memory, retrieval or head gating improves answers.

## Model, dependencies, and source pins

- Model: `LiquidAI/LFM3-1.7B`, revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
- Model SHA-256 (publisher LFS metadata and independently checked local bytes): `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`; 3,457,635,448 bytes.
- Config SHA-256: `8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d`.
- Modeling source SHA-256: `e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab`.
- Tokenizer JSON SHA-256: `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701`.
- `torch==2.14.1+cu130`, **CPU-only**; `transformers==5.21.0` in an isolated temporary package directory.
- `transformers==5.17.0` failed to load the official custom model config with `KeyError: 'conv'` in its RoPE configuration standardization; do not claim version compatibility before reproducing it.

## Exact reproducible source interface

`src/lfm3_checkpoint_attestor.py` requires `--offline-only --snapshot PATH` and verifies 11 pinned files, including safety checks for model-local and shared Hugging Face Xet cache symlink roots. The attestation does NOT execute model code or authenticate the publisher's web metadata.

`src/lfm3_real_checkpoint_smoke.py` requires:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.lfm3_real_checkpoint_smoke \
  --offline-only --ack-sandbox-research \
  --snapshot /approved/cache/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd \
  --gate-plan /approved/sg/gate_plan_0.8.json \
  --zero-plan /approved/sg/gate_plan_zero.json
```

**Do not execute that example directly in a privileged terminal.** Before invoking it, use a restricted, unprivileged, network-disabled container/namespace with only pinned model/cache paths and research code mounted **read-only**, an ephemeral writable `/tmp`, and no production `/nas` or SSH credentials visible. The runner checks for offline environment flags and the absence of two sensitive host paths, but those application-level checks alone do not establish isolation. The xwing run used `bubblewrap` user-level namespaces, an unshared network and PID namespace, read-only model/research mounts, and isolated `/tmp`.

The Rust plan was produced by the exact SensorGraft PR #7 head `ae96aac1d68119779aca3937cb88a2d7fbf57d38` using `sg_lfm3_gate_plan --synthetic-only 0.8`. The zero control used the same vector with `--synthetic-only 0.0`. The runner requires matching synthetic source fingerprint and model/config pin.

The Python runner verifies the actual checkpoint bytes before model load, uses the official local chat template, and captures last-token logits across four consecutive forwards: baseline, zero plan, nonzero plan, and restored baseline. It also snapshots all 11 gate projection weight tensors and compares their bytes in memory afterward. It emits no raw prompts, logits or trace payloads.

## Observed xwing result

See `research/results/lfm3_real_checkpoint_xwing_20261027.json`. In this one fixed 25-token prompt, all 11 attention hooks attached to 176 gate outputs; zero plan and restored baseline were bit-for-bit identical, the nonzero gate plan changed output logits, and projection weights remained unchanged. Maximum absolute last-token output-logit difference was **0.949219**. Four forwards were measured over **0.902 s** after model loading; load plus forwards **3.167 s**.

These are **not** tokens/sec, p95 throughput, or accuracy statistics; neither retrieval quality nor learned control was tested. The synthetic logit delta controller is still not a learned function of SensorGraft state.

## Remaining independent gates

1. Full GitHub CI: previously 1,105 tests passed with six skipped, but the repository's enforced total coverage requirement (69%) remained unmet at 56.76%. No coverage exception or bypass.
2. Build approved held-out graph/tool tasks with independently reviewed gold outputs and verified trace custody. Do not automatically treat historic trace metadata as learning labels.
3. Run the reproducible no-memory, retrieved-context, gate-only, and combined gates+retrieval conditions with identical prompts, decoding and model hashes. Report per-session paired wins/losses, latency, energy and raw task success.
4. Repeat on supported hardware/runtime combinations; use GGUF/Vulkan for deployment separately from this BF16 PyTorch hook.
5. Keep draft PRs unmerged and no production Neo4j/trace migration or model training until safety and rollback gates pass.
