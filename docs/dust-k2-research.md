# Dust zeroth-order K2-Horizon experiment — evidence-first gate

Tracking: [auto-finetune issue #1](https://github.com/scottjoyner/auto-finetune/issues/1)

## Scope and limits

This is **not** a new training backend. This PR supplies an opt-in, CPU-readable
preflight/provenance manifest and unit tests. It makes no model updates, invokes no
optimizer, and changes no scheduler, deployment, notification, authorization, or
retention settings. Preflight output always declares training_authorized=false.

Original work: [Dust research](https://qlabs.sh/research/dust),
[source](https://github.com/qlabs-eng/dust); expected upstream commit:
b20f7c03eac630b6441ba6f254128c1761dc45ad (2026-10-06).
The upstream minimal sample is a custom GPT **pretraining** implementation on
NVIDIA bfloat16 hardware. It does *not* offer an out-of-the-box K2-Horizon
fine-tuning entry point or established AMD/ROCm support.

Dust perturbs intermediate activations at independent token positions and
estimates local errors from loss differences. A successful 2-step upstream
sample would not establish correct K2 training, generalization, or efficiency.

## Live K2 preflight (read-only)

Use an existing, explicitly authorized host/session with local access to the
model. Never install the upstream CUDA requirements in the existing ROCm venv.
Do not copy sensitive corpus records into issues or public PRs.

~~~bash
/media/scott/data/finetune-venv/bin/python -m src.dust_preflight \
  --model-dir /media/scott/data/finetune-staging/models/K2-Horizon-0.9B \
  --adapter /media/scott/data/finetune-staging/outputs/checkpoints/k2-reflog-lora-4-v8-corrected \
  --upstream-dir /tmp/dust-upstream-research-20261006 \
  --probe-torch --strict-model \
  --output /tmp/k2-dust-preflight-new-unique-name.json
~~~

Add --sha256-weights for full model/adapter digests (reads the multi-GB shard);
use with headroom and a quiet I/O window. Add --dataset /path/train.jsonl
and --dataset /path/heldout.jsonl only for already-cleaned, known partitions.
Only hashes/counts are emitted, never examples or token contents.

The manifest reports shard existence, adapter identity, optional hashes,
ROCm/CUDA probe, source pin, and backing NAS mount. Output is created
exclusively; attempts to overwrite existing evidence are rejected.

## Confirmed initial risks and gates

- The Xwing local model directory contains the approximately 2.1 GB K2 shard;
  the observed NAS /nas/models/k2-horizon/model-code index listed one shard
  but the shard was not present. This path is NOT a verified model backup.
- Xwing has AMD Radeon 8050S on ROCm. PyTorch implements a CUDA-like
  device interface on ROCm; torch.cuda.is_available() is NOT proof that
  upstream CUDA/NVIDIA requirements are satisfied.
- Previously trained K2 LoRA artifacts v7 and v8 remain independent controls.
  Neither is presumed a successful held-out baseline without evidence.
- Before scheduling any forward loop, verify NAS backing device, available
  scratch memory, training occupancy/locks, and immutable source snapshots.
  NAS5 recovery/evacuation and output cleanup are out of scope.

## Next two research slices (not implemented here)

1. **Upstream reproduction:** pin the published Dust SHA; on compatible
   NVIDIA bfloat16 GPU, replicate the upstream two-step sample at population
   256 with fixed seed, fixed data and CPU/GPU utilization evidence. Otherwise
   first run *isolated* AMD ROCm API compatibility tests, never replacing the
   fleet's existing PyTorch/ROCm installation. Archive success *and* failure.
2. **K2 adapter port:** frozen K2 base, new rank-16 LoRA initialized identically
   to the matched PEFT backprop control, initially only attention o_proj.
   Implement token-local activation perturbation, estimated local error,
   LoRA-factor weight update, and masked held-out CE. No autograd backward call.
   q/k/v layers require Dust attention-specific credit assignment; do not
   assume the feed-forward estimator transfers unchanged.

Compare at fixed seeds, train/eval partition, chat template, token budget,
loss mask, and target modules. Measure wall-time/GPU-hours, loss, tool-call
validity, agentic completion, memory, and checkpoint integrity. Require
bounded-resource preflight, no scheduler contention, and a human-reviewed
separate PR before training or promoting outputs.

## Unit test

~~~bash
python -m pytest -o addopts='' -q tests/test_dust_preflight.py
~~~

All fixtures are tiny local synthetic files. Tests do not need a GPU, NAS,
model weights, Hugging Face downloads, or private session databases.

## Executed acceptance: Xwing, 2026-10-06 (Fleet Commander)

The actual Xwing K2 local base, v8 adapter, and NAS mirror were probed
without training, model mutation, downloads, or configuration changes.
The manifest was produced with the **existing ROCm virtualenv interpreter**.
Xwing's default system `python3` instead reports PyTorch 2.13.0+cu130 and
no available accelerator: selecting that interpreter gives a **false negative**
for the available AMD GPU. Do not use default `python3` for ROCm admission.

- Local K2 base: complete, 1/1 indexed shards; weight SHA-256
  `6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365`.
- Existing v8 LoRA: complete, rank 16, q/k/v/o projection targets; adapter
  SHA-256 `1acef422fc0f0fa8b596e36ff620055f5823770e641378a8ad39897768ccae40`.
- Existing held-out-combined split: 559 nonblank records; SHA-256
  `e19e15d2b3e300a84a4353dfa0b7a9954032afea3543a2eb17c28c94850fc854`.
- Existing ROCm venv: PyTorch 2.12.0+rocm7.14.0, HIP 7.14.60850,
  device `AMD Radeon 8050S Graphics`; upstream Dust compatibility unverified.
- NAS mirror: **expected fail, exit 2**; 0/1 K2 indexed shards found.
- `/nas` resolves to CIFS on Xwing; mount identity is recorded locally.
  This identifies a mounted path, **not** verified physical NAS5 custody.
- Full weight/dataset hashes above were measured live. These checks DO NOT
  validate model outputs, held-out separation, training success, or readiness.
