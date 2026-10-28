# LFM3 gated attention research harness — automatic fine-tuning boundary

2026-10-27 | Research only, experiment opt-in; **no production routing**.

Primary architecture: https://www.liquid.ai/blog/lfm3
Model card: https://huggingface.co/LiquidAI/LFM3-1.7B
Implementation: https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py

The official text model has 11 attention layers, each with
`self_attn.gate_proj: Linear(2048, 16)`. Its forward path computes
`sigmoid(gate_proj(hidden_states))` and scales head outputs before
`o_proj`. This module supports PyTorch forward hooks. A normal
OpenAI-compatible or GGUF inference endpoint does NOT expose this hook.

## Cross-repo experiment

1. Standalone SensorGraft Rust binary
   `sg_lfm3_gate_plan --synthetic-only STRENGTH` reads a bounded JSON
   float activation vector from stdin and emits 11 × 16 bounded
   gate-logit deltas; explicit synthetic-only plan, no raw traces.
2. auto-finetune `src/lfm3_head_gate_hook.py` validates **all** plan
   fields, head/layer map, delta bounds, source flags, and pinned model
   configuration/modeling-code fingerprints. `synthetic_head_gate_hooks`
   is disabled unless `enabled=True` and the exact model layout
   matches the pinned official LFM3-1.7B release. It attaches
   reversible hooks to the 11 `gate_proj` modules.
3. Inference then uses the model's normal
   `gate=sigmoid(gate_proj(hidden_states))` operation. The hook adds
   tiny per-head *logit biases* to the projection output, preserving
   all checkpoint parameters. On scope exit or failure, hooks are
   removed. Existing hooks are refused to prevent double application.

The plan is **per request**, static across tokens, and is NOT learned
or conditioned on individual-token hidden states. We have not proven
a retrieval quality improvement. No LoRA, kernel-level attention
editing, mixed RL, or model checkpoint inference is claimed.

## Exact model/source pins

- Repo revision: `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
- `config.json` SHA256:
  `8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d`.
- `modeling_lfm3.py` SHA256:
  `e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab`.

Do not execute unpinned remote model code in a privileged harness.
Any updated upstream revision requires a new explicit review and
repeated hook-shape tests. The pinned model may be used locally only
under the model's LFM Open License v1.0.

## Testing

Run SensorGraft's Rust tests first, then on a host with PyTorch and
cached official config/modeling files:

```bash
SG_LFM3_GATE_PLAN_BIN=/path/to/sensorgraft/target/debug/sg_lfm3_gate_plan \
LFM3_CONFIG_JSON=/path/to/official/config.json \
LFM3_MODELING_PY=/path/to/official/modeling_lfm3.py \
python3 -m unittest discover -s tests -p test_lfm3_head_gate_hook.py -v
```

Three hardware-independent CPU PyTorch tests on xwing passed:
normal injected forward response changed only by bounded target
deltas; original gate-proj weights remained unchanged; zero strength
matched baseline exactly; errors and context exit removed all hooks;
forged provenance or model source hashes were refused.

The xwing BF16 model weight file was **not** cached during testing;
this experiment used lightweight PyTorch structures matching LFM3
attention module shapes. No actual complete LFM3 inference, output
quality, or latency improvement has been demonstrated.

## Post-training implications

Liquid AI's published sequence is supervised fine-tuning (SFT),
mixed RL, six specialist RL teachers distilled via multi-domain
on-policy distillation (MOPD), and harness-grounded agentic RL.
We should not attempt to reproduce its total training budget.
Instead: curate *approved, de-identified and provenance-bound*
tool-success/failure pairs; train a small frozen-checkpoint head
controller or adapter with constrained logits; evaluate on held-out
sessions and compare against no-memory/retrieval-only/LoRA baselines.
Use offline domain specialists for graph retrieval, tool selection,
safety/abstention, and long-context. Introduce DPO/GRPO only with
reproducible checked rewards and sandboxed tools.

Real-time SensorGraft memory changes retrieval results, not LFM3
weights. A tiny trained head controller may steer activations under a
fixed model checkpoint, but must be versioned, measured and reversible.
Do NOT treat trace database rows or SafeTensors activation exports
as interchangeable with model parameter tensors.

## Fleet-specific next gates

- xwing Ryzen AI Max: evaluate updated llama.cpp GGUF under Vulkan,
  separate from PyTorch hook experiments. Avoid allocating the
  full BF16 checkpoint on the current 8 GB available-memory snapshot.
- x1-370: Rust control plane, trace index, generation custody and
  bounded activation plan generation.
- Joyner GPU: inspect installed ROCm driver, accessible VRAM and
  support for LFM3 before attempting BF16 adapter training.
- MacBook Air: evaluate MLX version separately when explicitly
  available, with 8 GB memory and model/context budgets accounted for.
- Every run: pinned revision/quantization, token prefill/decode,
  p50/p95/p99 latency, watt-hours, memory/RSS, failures, tool-call
  accuracy, abstention, isolation and rollback.

Do not merge or deploy based on synthetic shape tests alone.
