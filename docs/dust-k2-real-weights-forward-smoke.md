# Real K2-Horizon: bounded forward-only last-`o_proj` acceptance

Research issue: [#1](https://github.com/scottjoyner/auto-finetune/issues/1).
Stack on [PR #3](https://github.com/scottjoyner/auto-finetune/pull/3),
itself stacked on [PR #2](https://github.com/scottjoyner/auto-finetune/pull/2).

## What this slice proves

`experiments/dust/k2_forward_only.py` operates on the actual
`K2HorizonForCausalLM` module tree and optionally its pretrained
K2-Horizon-0.9B weights. It registers a temporary hook on the
**final decoder layer**'s attention `o_proj`, attaches newly initialized
rank-4 LoRA factors, and estimates the token-position-local loss derivative
through activation Gaussian noise. It applies a direct tensor update to
LoRA factors A and B under `torch.no_grad()`, with no optimizer or
`backward()`. No weight/adapter/dataset/checkpoint files are modified.

The **final layer** matters: K2's decoder has no subsequent attention
after this projection, so its position-t output affects only position-t
logits through final-layer residual/MLP/normalization/head operations.
This makes the position-local causal CE reward appropriate for this
specific probe. Earlier layers and Q/K/V projections require attention
future-token credit assignment; do not extrapolate this estimator.

The probe shifts `labels[:,1:]` onto `logits[:,:-1]`, ignores masked
prefix labels (`-100`), and never uses outputs without an eligible target.
Only synthetic, deterministic token IDs are used. This is **not** an
evaluation of model usefulness or genuine fine-tuning on curated data.

## Guardrails

- Default: randomly initialized 2-layer **tiny K2 architecture** (256-token
  vocab, 64 hidden units). Real pretrained weights require `--pretrained`
  and use **CPU only**; ROCm has no approved full-model memory budget.
- One step / at most four draws for real K2; tiny models allow two steps /
  128 draws. Rank restricted to 2/4/8. Requires >=4 GiB available memory
  before loading pretrained weights and >=1 GiB at each forward call.
- Existing ROCm PyTorch and local Hugging Face code; no network downloads,
  package installation or change to fleet scheduler. The model has
  `requires_grad=False` throughout and the original `o_proj.weight`
  is checked bit-for-bit after adapter-only updating.
- Hooks are removed in `finally`. Default output is stdout JSON,
  and an explicitly selected `--output` exclusively creates metadata,
  refusing an existing path before loading weights.
- No real held-out dataset, no implicit merge/deploy, and no NAS writes.
  Do not automate dispatch under the existing training watchdog.

## Reproduce (authorized Xwing only)

~~~bash
# Tiny K2 architecture on AMD ROCm, no pretrained weights:
timeout 75 /media/scott/data/finetune-venv/bin/python \
  experiments/dust/k2_forward_only.py \
  --model-dir /media/scott/data/finetune-staging/models/K2-Horizon-0.9B \
  --device cuda --draws 16 --steps 1 --seed 42

# Pretrained K2 base, CPU-only, one step / at most four draws:
# First verify available RAM and ensure no other training occupies host.
flock -n /tmp/dust-k2-last-o-proj-smoke.lock \
  timeout --signal=TERM --kill-after=5 120 \
  nice -n 10 /media/scott/data/finetune-venv/bin/python \
  experiments/dust/k2_forward_only.py \
  --model-dir /media/scott/data/finetune-staging/models/K2-Horizon-0.9B \
  --pretrained --device cpu --draws 4 --steps 1 --seed 42 --lr 0.1
~~~

## Executed evidence — Xwing via Fleet Commander, 2026-10-06 local

Environment: Xwing AMD Radeon 8050S; PyTorch 2.12.0+rocm7.14.0
with HIP 7.14.60850. The tiny K2 architecture runs on AMD GPU with
16 activation draws, one step, seed 42, and decreased synthetic token
cross-entropy from **5.45205 to 5.40390** in approximately 1.35 seconds.
Model-projection weights stayed unchanged, no backward call,
no gradients, no checkpoint writes.

**Real pretrained K2 weights were loaded successfully on CPU** from the
local 0.9B checkpoint. The last-layer `o_proj` hook and LoRA estimator
completed one bounded step with short synthetic tokens, confirming
architecture and runtime compatibility. A first 2-draw trial at `lr=1`
**increased** synthetic sequence loss from 7.10383 to 7.12605;
that is a valid negative result, not a success claim.

A subsequent three-seed / four-draw trial at `lr=0.1` gave:

| Seed | Pretrained K2 synthetic-token CE before | After | Improved? | Time for inner step |
| --- | ---: | ---: | :---: | ---: |
| 7 | 7.103835 | 7.099339 | Yes | 1.050 s |
| 42 | 7.103835 | 7.106124 | **No** | 0.922 s |
| 1337 | 7.103835 | 7.094564 | Yes | 0.984 s |

These are **training-sequence** losses on synthetic token IDs, not
held-out CE. Small populations are noisy and improvement is not
reliable, even though the adapter changed and pretrained model did
not. Three seeds are not a statistically meaningful quality evaluation.

The isolated test suite on Xwing's ROCm Python passed **5/5**,
including exact causal-loss shift/masking, parameter admission gates,
actual tiny K2 forward without backward, and exclusive-create evidence
semantics. Upstream Dust was **not** reproduced; our estimator is
Dust-inspired and adapted specifically to last-layer causal CE.

## Next gate

Before claiming fine-tuning, establish a **frozen, immutable train/eval
split** without leakage, run a matched `o_proj` backprop-LoRA control,
and compare held-out token CE/quality, GPU-hours and memory across seeds
at a meaningful perturbation population. Then implement the
attention-specific causal future-credit assignment needed for any
earlier-layer `o_proj` and for Q/K/V targets. Require separate
reviewed compute admission and checkpoint custody before launching
larger multi-step pretrained training; do not merge this prototype
into the existing `src.train` backend automatically.