# E1c — official LittleBit CPU initializer and checkpoint parity

**Date:** 2026-10-09 | **Stage:** E1 official-class CPU smoke | **Status:** measured, draft research review; E2 QAT remains HOLD.

## Reproducible source and environment

- Official code: `SamsungLabs/LittleBit@42d658b0c79f76450b34b6a3547462c7cdc3e1a0`.
- Exact verified Git source blobs: `littlebit.py=43ce9d2f9383b676c3898346ece2d3994269a908`, `binary_packer.py=78f0e20f8a525d3cdb0f94908fd767077834cf4b`.
- Host: x1-370 (CPU-only), Python 3.12.3, **torch 2.6.0+cpu** in an isolated virtual environment `/home/scott/git/littlebit-parity-venv-20261009`; installed via `uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'` without modifying system Python.
- Executed **the actual pinned upstream `LittleBitLinear` class, `__quant_convert__`, `state_dict()`, factorized `forward()`, and upstream `binary_packer`** on synthetic 64×64 rank8, 128×256 rank16 and 256×128 rank16 matrices.
- The independent harness imports only two pinned upstream files into an isolated namespace, checks their blob hashes and Git HEAD, and opts in via `--execute-cpu`. It does not vendor upstream code, import full training stack, contact hosted providers or invoke production/scheduler paths.
- Uses `quant_func=torch.sign` as an explicit *nontraining binary-sign forward control*, not a reproduction of the upstream STE/SmoothSign gradient. No autograd/QAT update was run.
- Reseeds the official randomized low-rank SVD for paired controls. Importantly, Joint-ITQ itself consumes random draws, so subsequent upstream randomized rank-one SVD can see different RNG state; this is an **official recipe-level contrast**, not a causal proof isolating rotation alone.

## Official-class CPU experiments

Seeds 7/42/1337; three shapes; 9 matched pairs per residual flag. For every pair, both initializers see identical source weight tensors (SHA-256 matched). Below, delta = Joint-ITQ minus SVD-only **relative Frobenius reconstruction error** (lower is better).

| Variant | ITQ iterations | Pairs | Joint-ITQ better | Mean delta | SHA-256 of complete JSON observations |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary only | 5 | 9 | 9/9 | −0.0083802475 | `8549bca07f4c1bda84f6919bcaad2fed6aa131f70a6fb10320bc99051747e712` |
| Primary + residual | 5 | 9 | 9/9 | −0.0179908143 | `6b60b14ab1a99cff4e898127c2f446e3265e31aae6bdc11a34f66fed7cb25dcb` |
| Primary only | **50** | 9 | **9/9** | **−0.0130507681** | `e8bb5d05d1c4d75ae0e85ea60a82cadb83038d25f040196ffbdabc0105dde29c` |
| Primary + residual | **50** | 9 | **9/9** | **−0.0248236524** | `8cfd0e18947611e64ff39a3a795a83d4287a5962a9f268fa8d1dcd094bb89e29` |

These are **exploratory synthetic data observations**, not held-out CE, model perplexity, generalization, and not a sample-size-controlled statistical claim. The 50-step run was an *extension* after viewing 5-step results, not an independently preregistered confirmatory study. It used the class's documented default ITQ iteration count.

## Actual official-class state_dict storage

For the pinned source's **float32 scale path**, tensor BPW counts real `state_dict()` tensor bytes, including FP32 `u1,u2,v1,v2`, row-padded packed int32 sign-factor tensors, int64 shape tensors and three buffers. ZIP BPW is measured by actual `torch.save(state_dict, BytesIO)` on **one layer**; it must never be advertised as whole-model BPW.

| Shape | Rank | Residual | `_compute_eff_bits` BPW | **Actual state tensors BPW** | One-layer ZIP BPW |
| --- | ---: | --- | ---: | ---: | ---: |
| 64×64 | 8 | no | .781250 | **1.843750** | 8.853516 |
| 128×256 | 16 | no | .382812 | **.667969** | 1.528564 |
| 256×128 | 16 | no | .382812 | **.730469** | 1.591064 |
| 64×64 | 8 | yes | 1.562500 | **3.656250** | 14.306641 |
| 128×256 | 16 | yes | .765625 | **1.332031** | 2.632080 |
| 256×128 | 16 | yes | .765625 | **1.457031** | 2.757080 |

The discrepancy is driven by actual **FP32**, rather than nominal FP16, scale storage, a missing rank-length scale in the upstream estimator, int32 row padding, tensor-shape values and registered buffers. Upstream efficiency assumptions may differ after explicit half-precision casting or conversion for deployed checkpoints; investigate those paths separately before reporting errors in the published paper's original effective-BPW formula.

All tested dense-vs-factorized `forward` comparisons passed the harness threshold of 1e−3 absolute error. Any file-archive comparison needs to control per-file ZIP overhead and report actual whole-model bytes.

## Exact research commands

In a separate CPU-only environment with pinned source checked out at that exact commit, from the isolated research worktree:

```bash
LITTLEBIT_UPSTREAM_ROOT=/home/scott/git/littlebit-upstream-audit-20261009 \
  python -m unittest discover -s tests -p test_littlebit_official_class_parity.py -v

python -m experiments.littlebit.official_class_parity \
  --upstream-root /home/scott/git/littlebit-upstream-audit-20261009 \
  --execute-cpu --itq-iterations 50

python -m experiments.littlebit.official_class_parity \
  --upstream-root /home/scott/git/littlebit-upstream-audit-20261009 \
  --execute-cpu --residual --itq-iterations 50
```

On x1, the actual executable is `/home/scott/git/littlebit-parity-venv-20261009/bin/python`. The four full aggregate JSON manifests (above hashes) are retained as local **research-only artifacts** under `/home/scott/git/littlebit-upstream-*-20261009.json`; do not claim GitHub contains their full bytes.

## Acceptance verdict and next gate

**E1 real official CPU initialization + state_dict serialization: PASS (synthetic, bounded).** Nine focused regression tests passed on x1, including exact pinned source, opt-in, source-tamper rejection, bad revision, range limits, deterministic repeat, scales dtype, pack keys, residual and forward parity. This is a standalone research smoke; independent CI and code review are still required.

**NOT_RUN / HOLD:** official pretrained Qwen block conversion, actual BF16/FP16 inference serialization, full-model bytes, teacher/student KL, scales-only QAT, held-out model CE, inference kernels, AMD fleet portability and all production deployment.

**Recommended next experiment:** read-only compatibility check on a pinned public Qwen3-0.6B revision, with explicit license, local storage, memory and resource approvals. Before any model load, lock layer map, quantized module selection, rank ceiling, candidate target BPW under FP32/FP16/BF16 serialized variants, and contamination-free heldout protocol. Continue to prohibit scheduler/merge/deploy paths.
