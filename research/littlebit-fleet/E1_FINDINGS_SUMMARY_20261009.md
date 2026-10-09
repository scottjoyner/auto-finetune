# E1 CPU initializer research findings — 2026-10-09

**Status: independently implemented synthetic CPU evidence; upstream parity and real-model QAT NOT RUN.** Tracking: #17, project design PR #18.

## Sources and method
Inspected `SamsungLabs/LittleBit@42d658b0c79f76450b34b6a3547462c7cdc3e1a0`, especially `quantization/modules/littlebit.py` and `quantization/utils/binary_packer.py`. An isolated NumPy reference compares deterministic full-SVD-only initialization, seeded random orthogonal rotation, and 30-step alternating Joint-ITQ Procrustes. **This does not execute or reproduce upstream `torch.svd_lowrank`.** All weights are seeded synthetic matrices; no model checkpoint was read.

## Findings and limits
- Original exploratory pilot: 84 matched pairs, seeds 7/42/1337, three layer shapes and four spectral families, rank multiple-of-8; Joint-ITQ's relative Frobenius reconstruction error was lower than SVD-only in 84/84 and lower than random rotation in 82/84.
- New-fixture storage-gated check: 48 matched pairs, **new seeds 20261/20262/20263**, three shapes and four spectral families; Joint-ITQ outperformed SVD-only **48/48** and random orthogonal rotation **48/48**. Mean ITQ-minus-SVD relative-Frobenius-error delta **-0.11902**, mean ITQ-minus-random delta **-0.04534**. Largest average difference occurs in high-coherence heavy-tailed matrices. Pilot-informed test design means **these are exploratory synthetic findings**, not claims of statistical generalization.
- 26 CPU-focused tests passed; 11 original standard-library oracle/packing tests, 13 new NumPy tests and 2 optional independent Torch CPU equation checks. No upstream model/QAT test or GPU job passed or was attempted.
- Important correction: upstream persists `u1,u2,v1,v2` scales (two rank-length scales); its `_compute_eff_bits` formula appears to omit **16 × rank bits per branch** when FP16. The upstream packer also stores **int32 words padded per row**, so its tensor payload BPW can exceed theoretical logical BPW. Example 64×64 rank8: logical 0.8125 BPW, upstream-reported formula 0.78125, *minimal upstream-style tensor payload* 1.25 BPW (including scale and shape tensors, excluding file container).
- New-shape physical storage gates prohibit a 0.30 BPW primary branch for 128×256 and 256×128 under the tested rank-multiple-8 setting. This is an **infeasible configuration**, not a failed model quality result.
- The first attempted pilot hit a zero-sign ambiguity on degenerate spectra, with no successful measurement. Fixture degeneracy was removed and documented before the accepted runs.

## Provenance / artifact handoff
Independent CPU bundle is available as a conversation artifact and is **not yet in this GitHub branch**. The earlier x1 CPU oracle remains local at commit `45b4dd3cfbc8e77f09dae4df6d39270508409bd3`: its git push failed due to missing credentials; connected file transfer was safety-blocked and not overridden.

Recorded checksums:
- Fresh-seed protocol SHA256 `cf38c6327e410790151979f802d1bb752c1815c36a9780daef39ede41a4bd02a`
- Pilot JSON SHA256 `278ec06bd6b0c75b9b6f4bc61b356c1296096a6e0b9cf59d0f29aca8bffb80ae`
- New-seed JSON SHA256 `51a5a3ff24d4d78616d9201a6cca14f5f4c4fda443bd446eab4d585b581d6c79`
- Test output SHA256 `e3422b07bbef1eaafac1fca9a13041fe235f49b3c029fa84877599927b8f994a`
- Source/evidence bundle SHA256 `f83104cde8b221aa08450cfaa7a05ed42e18148d0bd5daedea56fecbc313726d`

## Gate
**E1 oracle + independent initializer reference: positive exploratory outcome.**
**E1 full upstream implementation parity: HOLD.** Need a compatible pinned CPU Torch environment, official initializer comparison on identical matrices, actual checkpoint serialization byte audit and rights/dependency review. **E2 pretrained model/QAT/GPU: HOLD.** No deployment, scheduler, provider or NAS write authorization.
