# Dust-inspired activation-noise LoRA: synthetic ROCm sanity probe

Tracks [research issue #1](https://github.com/scottjoyner/auto-finetune/issues/1).
Stacked on the **read-only** K2/Dust provenance PR #2. This experiment does
**not** train, load, or evaluate K2-Horizon or reproduce upstream Dust's full
transformer experiment.

## Scope

`experiments/dust/zo_lora_probe.py` builds a tiny, entirely synthetic
tokenwise linear regression problem with a frozen base matrix and rank-4
LoRA factors `A` and `B`. It estimates per-token activation-output errors
from Gaussian perturbations, maps those estimates into LoRA-factor updates
using cached inputs, and performs updates with explicit tensor additions.

For this probe, the loss is tokenwise squared error, **not** language-model
cross entropy. The teacher's correction lies exactly in the LoRA span;
improvement on these synthetic samples is a numerical sanity check only.
The gradient cosine reported is against the exact **analytic** factor-`B`
gradient (not a reference autograd backward pass). No `backward()`,
`autograd.grad()` or gradient tape is used; model weights have
`requires_grad=False`.

This is *Dust-inspired* zeroth-order activation perturbation, not a faithful
K2 architecture port, not an endorsement of compute efficiency, and not
evidence of held-out generalization.

## Safe execution

The command creates no checkpoint, model cache, dataset, network request,
GPU service, scheduler change or NAS output. Default output is stdout JSON.
RAM use is bounded by a 128-token toy batch and at most 512 perturbations.

~~~bash
# CPU / installed torch
python experiments/dust/zo_lora_probe.py --device cpu --draws 128 --steps 6

# Xwing AMD GPU, using the existing ROCm virtualenv
/media/scott/data/finetune-venv/bin/python \
  experiments/dust/zo_lora_probe.py --device cuda --draws 128 --steps 6

# Optional: create an immutable NEW JSON evidence file (never overwritten)
python experiments/dust/zo_lora_probe.py --output /tmp/new-dust-probe.json
~~~

## Executed Xwing ROCm evidence (2026-10-06 local / 2026-10-07 UTC)

Executed through Fleet Commander over SSH with the existing
`torch 2.12.0+rocm7.14.0` interpreter, on AMD Radeon 8050S.
Six forward-only updates, 128 independent activation draws per step,
`sigma=0.05`, `lr=2`; elapsed time excludes interpreter startup.

| Seed | Initial loss | Final loss | Minimum cosine to exact B gradient | GPU kernel time |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.61853 | 0.00844 | 0.9947 | 0.338 s |
| 42 | 0.65963 | 0.01580 | 0.9962 | 0.335 s |
| 1337 | 0.64777 | 0.01175 | 0.9959 | 0.350 s |

These losses are on **the same synthetic training samples**. All three
runs reported `backward_calls=0` and `checkpoint_written=false`.
ROCm logged a `rocSHMEM Could not open libnuma` warning, but the
synthetic probes finished without runtime failure.

Four isolated Python/pytest tests passed on Xwing's ROCm environment:
deterministic fixed seed, bounded hyperparameters, refusal to overwrite
evidence, and a guard that makes `torch.Tensor.backward`,
`torch.autograd.backward`, and `torch.autograd.grad` raise if called.
On x1-370 (without torch in system Python), the toy tests skip and
the baseline read-only preflight suite still passes.

## Next gate: upstream Dust / real K2 compatibility

This check validates the **primitive only**, not the upstream Dust
system or an LLM fine-tuning recipe. Before real K2 training, require:

1. A pinned, isolated upstream reproduction on compatible hardware;
   or inspect the NVIDIA-specific dependencies and port the needed API
   operations to ROCm without altering the existing training venv.
2. A frozen pretrained K2 base plus a fresh, independently validated
   LoRA on `o_proj`. Correct causal token-loss masking and tensor shapes
   must be verified before adding `q_proj`, `k_proj`, or `v_proj`;
   these require attention-specific causal credit assignment.
3. Identical train/held-out split, seed, template, and GPU/time budgets
   against a matched conventional LoRA run. Report full GPU-hours and
   held-out outcomes; do not infer performance from this toy loss.
4. A new, explicitly reviewed runtime gate with GPU/host locks,
   bounded resource use, verified NAS custody, and no auto-promotion.

Existing K2 v7/v8 weights, auto-finetune schedule, and NAS recovery
processes remain unchanged by this probe.
