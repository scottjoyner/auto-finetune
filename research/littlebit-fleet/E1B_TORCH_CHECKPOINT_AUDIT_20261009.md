# E1b — real CPU PyTorch tensor / checkpoint-framing audit

**2026-10-09 | Status: TESTED, BUT UPSTREAM CLASS PARITY STILL HOLD**

**Project:** Issue #17; design PR #18. Previous E1 synthetic reference and Joint-ITQ comparison remain independently documented. This note adds a *storage-fidelity* control, not a real model result.

## Pin / custody

Inspected the official repository at `SamsungLabs/LittleBit@42d658b0c79f76450b34b6a3547462c7cdc3e1a0`. On x1-370, an isolated read-only checkout produced matching Git blob IDs:
- `quantization/modules/littlebit.py` — `43ce9d2f9383b676c3898346ece2d3994269a908`
- `quantization/utils/binary_packer.py` — `78f0e20f8a525d3cdb0f94908fd767077834cf4b`

Confirmed source definitions for `_compute_eff_bits`, `_decompose_matrix`, `_rank_one_decompose`, `_compute_itq_rotation`, `state_dict`, and `pack_weights`; no official class execution was performed (the x1 research environment did not have a suitable Torch installation).

## Distinctions that change the experiment

1. For each branch, `LittleBitLinear` has **four scale tensors**: `u1[out]`, `u2[rank]`, `v1[rank]`, `v2[in]`. Our earlier three-vector synthetic factor oracle fused the two rank-length scales. A fused inference algebra can be equivalent but its storage must not be confused with the **upstream checkpoint format**.
2. Upstream `_compute_eff_bits` appears to omit **one rank-length scale** — 16 × rank extra bits per branch assuming FP16. It also does not account for int32 packed row padding or shape tensor storage.
3. Upstream packer pads each factor-matrix row separately to a 32-bit word boundary. `state_dict` retains the scale tensors and buffer metadata and includes packed sign factors + int64 shape tensors.
4. Small one-layer ZIP archives have large framing overhead; the whole-model archive may amortize that overhead. Only *real complete checkpoint serialization* can support an end-to-end BPW claim.

## Observed independent Torch audit

A research-only Python module built an **upstream-shaped state dictionary**, exercised a separately implemented canonical sign packer, counted physical PyTorch tensor bytes, and called real CPU `torch.save` to `BytesIO`. It did **not** import the official class or load actual trained weights.

`31/31` focused tests passed in Python 3.13.5, `torch==2.10.0+cpu`, no CUDA. This extends the previous 26 E1 tests by five serializer/padding/metadata checks. No model download, GPU run, private data, provider calls, production services, NAS write or training occurred.

| Shape | Rank | Branches | Upstream formula BPW | Correct logical FP16 BPW | Estimated sign+scale+shape tensor BPW | Actual synthetic tensor BPW incl. buffers | Standalone saved-ZIP BPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64×64 | 8 | 1 | .781250 | .812500 | 1.250000 | 1.281250 | 8.753906 |
| 128×256 | 16 | 1 | .382812 | .390625 | .460938 | .464844 | 1.391113 |
| 128×256 | 24 | 1 | .480469 | .492188 | .531250 | .535156 | 1.453613 |
| 256×128 | 16 | 1 | .382812 | .390625 | .523438 | .527344 | 1.453613 |
| 128×256 | 16 | 2 | .765625 | .781250 | .921875 | .925781 | 2.322754 |
| 4096×4096 | 32 | 1 | .023468 | .023499 | .023514 | .023521 | .025300 |

**Clarification:** Actual torch tensor BPW here includes three small upstream-shaped metadata buffers (16 bytes total) and uses FP16 scale dtype by assumption. Standalone ZIP BPW is a measurement of a simulated *one-layer* torch.save dictionary, not a deployed model checkpoint; the figures must not be represented as actual official `LittleBitLinear.state_dict` output.

Evidence SHA256:
- Test output: `f65ee78e64729492d5c4009df23a5d1e80a000303cf14ad17c2877685a1e269b`
- Six-row Torch audit JSON: `f11a551b40eeae62180781242556f071622f21750ffb3ab82c662cb03f9f8466`
- Self-contained research bundle SHA256: `35df94f6cc9235f239755a6e3d454132e73bd1800be883a23445212d0f773ab7`

Research bundle supplied in the associated ChatGPT conversation; no code has been pushed from the sandbox to GitHub.

## Remaining exact gate (do not infer a pass)

- Run **pinned official** `LittleBitLinear` CPU initializer + packed `state_dict()` in a safe dependency-pinned research venv, compare signed tensors and FP16/BF16 scales, and assert actual saved bytes and `eff_bit_actual` computed metadata.
- Compare official `torch.svd_lowrank` randomized behavior at controlled torch RNG seeds with deterministic independent full-SVD reference. Don't claim numerical equality for different SVD implementations.
- Lock model/data revisions and license rights before any pretrained Qwen layer experiment.
- GPU/QAT/inference/production remain **HOLD**.
