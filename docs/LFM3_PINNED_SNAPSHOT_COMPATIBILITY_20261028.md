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