# LFM3 + SensorGraft + Trace Viewer: versioned research integration

October 27, 2026. This adapter is metadata-only, opt-in,
loopback-only and **not a trained LoRA, weight patch or deployed LFM3**.

Liquid AI released the LFM3 ~2B family, including LFM3-1.7B text,
VL-2B, Audio-2B, and Omni-2B. Confirm exact model revisions,
supported inference runtimes, LFM license, tokenizer/tool-call behavior,
and adapter module names from the official model card before training
or distributing a model. Model identifier must be pinned to a
resolved checkpoint digest for any benchmark, not only a mutable name.

## Three independently versioned products

- SensorGraft: Rust graph/vector context index with immutable segment
  generations, native trace metadata and optional activation overlays.
- Trace Viewer: separately versioned private browser UI/API and
  categorical read-only trace-memory.v1 evidence endpoint.
- auto-finetune: safe consumer and, **later**, offline curated training
  exporter + independently benchmarked LFM3 LoRA workflow.

## Two different augmentation paths

**Online retrieval conditioning (first):**
A caller explicitly opts into a loopback Trace Viewer retrieval.
The 8192-byte bounded evidence packet contains categorical day,
status, source, kind and rowid only. The adapter validates every field
and creates separate non-instructional context messages. This provides
NO validated causal remediation and NO semantic relevance guarantee.

**Offline weight training (later):**
A separate approved pipeline curates/redacts permitted trace examples,
deduplicates, partitions by session/date, constructs labeled
input/output pairs, and trains PEFT/LoRA against a pinned LFM3
checkpoint. Evaluate contamination, privacy, safety, tool-use,
regressions, throughput and quantization before producing a signed,
revocable candidate adapter. Training and promotion must never happen
automatically from raw traces or production traffic.

A third path, SensorGraft's SafeTensors context embeddings, represents
**activation/context tensors**, not LFM3 layer weights. There is no
verified mapping into LFM3 internal state or live PEFT loader today.

## Explicit use, no hidden model calls

Import src.trace_memory_bridge and call fetch_local_metadata with
enabled=True, status='error', limit=8, base_url set to an authorized
numeric loopback Trace Viewer. The viewer must be started locally with
TRACE_VIEWER_ENABLE_MEMORY=1. Then call compose_inference_messages
with your prompt and validated evidence. The result is ordinary
OpenAI-compatible chat messages, not inference execution.

## Gates to real LFM3 integration

1. Confirm model-card license and exact revision, runtime
   compatibility and permitted local hardware.
2. Add signed/revocable SensorGraft generation plus an authenticated,
   least-privileged viewer retrieval API.
3. Add reproducible offline evaluations versus no-memory and
   unspecialized LFM3, using traced failure/resolution evidence.
4. Only promote separately trained LoRA candidates after audit and
   model-weight integrity checks. No automatic agent/model takeover.

No trace database or raw personal data is committed to public GitHub.