# LFM3-1.7B pinned checkpoint and xwing compatibility gate

**Research / no automatic promotion.** Reference revision:
`LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
Pinned SHA256 manifests are in `src/lfm3_pretrained_guard.py`.

## Read-only xwing evidence (2026-10-28 EDT)

- Local weight file: `model.safetensors`, 3,457,635,448 bytes; SHA256
  `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
  Hugging Face's pinned-revision resolve response reports the same linked digest.
- Local `modeling_lfm3.py` SHA256:
  `494631b2d3226c0feaa708f31ef8f479733ba6ff2253aa5232604b8525c3c742`.
  Official source SHA256:
  `e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab`.
- Local `config.json` includes an extra `rope_parameters.conv=null`;
  the official config has no such entry.
- Local `configuration_lfm3.py` is a standalone config object without
  `from_pretrained`. Local `Lfm3ForCausalLM` extends a base class inheriting
  directly from `torch.nn.Module`, also without `from_pretrained`.
  **Therefore a library `from_pretrained` call is not a valid substitute
  for actually testing the local patched implementation.**
- Current xwing has live inference services; do not stop them or start
  heavy model loading merely to clear this research gate.

## Required validation order

1. Preserve the local modified files; never overwrite them in place.
2. In an isolated directory, stage the official pinned revision with
   verified model, config, modeling code, tokenizer and all relevant hashes.
3. Identify and review any compatibility adaptations needed for the selected
   Transformers/runtime version. Re-pin every approved modified file explicitly
   to a distinct revision manifest; don't silently accept a modified module.
4. Confirm that a compatible loader exposes the requested APIs and checks
   **all** expected tensor names, dtypes and shapes; no random initialization,
   no missing/unexpected model parameters.
5. Run a bounded, independent real-checkpoint load and inference baseline on
   an admitted node without disrupting resident services.
6. Only then run paired AdamW/Muon optimizer experiments on matched,
   disjoint data and multiple seeds. Use the read-only SensorGraft observer;
   any automatic weight promotion remains disabled.

The snapshot check reads data but performs **no training**, no network
retrieval and no service mutation. It is intentionally fail-closed on the
currently modified xwing snapshot. The local digest manifest is not a
cryptographic signature of the model publisher; the remote reference and
release provenance must be independently reviewed.

## Verified metadata-only tensor audit

The new `src/lfm3_tensor_manifest.py` inspects the SafeTensors header only, never
allocating model weights. It rejects duplicate JSON keys, malformed shapes,
non-BF16 tensors, overlapping/out-of-bounds byte spans, missing model layers,
missing gate matrices, and unexpected tensor/parameter counts. The official
snapshot gate runs this check after validating SHA-256 digests. Verified read-only
against xwing's actual 3.46 GB model file: 286 BF16 tensor entries, 24 model
layers, 11 attention gate tensors, 1,728,801,536 elements, with all 3,457,603,072
tensor-data bytes accounted for. This does **not** establish compatibility with
the modified local model source or a successful pretrained inference load.

## Verified pinned source staging and CPU-only loading (October 28, 2026)

Hugging Face's `/raw/` response for the large `tokenizer.json` is a Git LFS
pointer, not the tokenizer payload. The corrected SHA256 is
`7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701`,
confirmed against the revision-pinned `/resolve/` linked digest and both
fleet nodes' cached payload. The previous pointer digest must not be used.

On xwing, the official-revision files were staged in a separate private
research directory using cached source files and a **hard link** to the
already verified 3.46 GB weights; resident inference was untouched. The
snapshot and tensor-manifest checks passed. Configuration and model-class
imports passed in a separate Python process with Transformers 5.17.0 and
the existing `src.lfm3_support.patch_transformers_for_lfm3`.

On x1-370, the existing `/home/scott/lfm3-research/model-metadata` snapshot
passed all pinned hashes and tensor checks. A standalone Transformers 5.17.0
+ tokenizers 0.23.3 overlay was installed in
`/home/scott/lfm3-research/isolated-transformers-5.17.0`, without mutating
the existing ROCm environment (PyTorch 2.9.1+rocm6.4).

**Real pretrained loading passed** with `AutoConfig` and
`AutoModelForCausalLM`, entirely CPU-only, using
`output_loading_info=True`. Transformers 5.17.0 returns empty **sets** for
some loading-report fields rather than lists; the guard accepts only empty
list/tuple/set collections and still refuses missing or nonempty fields.

One forward-compatibility issue was discovered and resolved in-memory:
Transformers remaps the official `conv` layer labels to `linear_attention`,
while the official pinned LFM3 model's mask builder requires `conv`.
The guard restores the checkpoint's original 24 layer labels **only when**
the observed runtime labels exactly equal that known transformation; any
different remap fails closed.

A verified four-token CPU-only inference forward then returned finite logits
of shape `(1, 4, 128000)` in 0.357 seconds (single trial). The final loader
has **33/33 focused synthetic tests passing**. No service, model file,
LoRA adapter, optimizer, or production setting was changed.

This demonstrates loading and one tiny functional smoke test, **not** model
quality, sustained throughput, training convergence, or autonomous weight
improvement. Multi-seed held-out evaluation and AdamW/Muon parameter grouping
remain independent, outstanding gates.