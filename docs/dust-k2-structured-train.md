# K2-Horizon orthogonal-antithetic forward-only LoRA training fidelity

This research slice stacks on the gradient-calibration work and remains
isolated from `src.train`, the scheduler, deployment, checkpoint promotion
and NAS recovery.

## Goal

Gradient calibration showed that the original K=64 one-sided Gaussian Dust
pilot was too noisy, while a full 1536-direction orthogonal antithetic basis
nearly recovered the exact local last-`o_proj` gradient.

The next question is stricter: **does that gradient fidelity survive actual
multi-step LoRA training after both A and B factors begin changing?**

`experiments/dust/k2_structured_train_compare.py` compares:

- a fresh rank-4 last-layer `o_proj` LoRA trained with exact adapter-only
  `torch.autograd.grad`;
- an identically initialized fresh rank-4 LoRA trained with randomized
  orthogonal, antithetic activation perturbations and **no backward call**.

Both use the same pretrained K2-Horizon-0.9B BF16 base, same two training
examples in the same order, same learning rate, same deterministic cleaned
dataset selection and the same prompt-disjoint heldout slice. The original
base model remains frozen and no adapter checkpoint is saved.

Two update steps are intentional: at initialization B=0, so the first exact
A-gradient is zero. The second update proves whether the forward-only method
can also track the A update after B has become nonzero.

## Live Xwing results — seed 42

All runs used sigma=0.25, lr=0.1, 8 selected training examples for aggregate
before/after CE, 6 selected heldout examples, but performed exactly **two
updates** using the first two selected train examples.

| Population K | A-update cosine vs backprop | B-update cosine vs backprop | Forward-only train CE delta | Forward-only heldout CE delta | Forward-only time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.32584 | 0.60441 | +0.000083 | +0.000325 | 88.16 s |
| 1024 | **0.93034** | **0.81621** | **-0.000936** | +0.000429 | 177.29 s |
| 1536 | **0.99796** | **0.99144** | **-0.000646** | +0.000320 | 262.32 s |

The matched backprop control used the same initial adapter and examples:

- train CE delta: **-0.000701**
- heldout CE delta: **+0.000274**
- runtime: roughly 4 seconds in these runs.

### Strongest evidence

At the full K=1536 orthogonal basis, after two real K2-Horizon updates:

- LoRA-A update cosine to exact backprop: **0.99796**
- LoRA-B update cosine to exact backprop: **0.99144**
- A update norm ratio: ~1.006
- B update norm ratio: ~1.007
- forward-only train CE change closely matched the backprop control.

That is an end-to-end confirmation that, for this constrained final-layer
LoRA case, the backpropagation-free update path can reproduce conventional
adapter training extremely closely when supplied a complete structured
activation basis.

## Important negative result remains

Neither method improved the tiny heldout CE after two updates. The full-basis
result establishes **update fidelity**, not model-quality improvement.

The compute cost is severe. K=1536 requires two perturbed forwards per
direction per step, making it roughly tens of times slower than the matched
adapter-only backward control. K=1024 is a more interesting compromise:
A-update cosine ~0.93 and B-update cosine ~0.82 while reducing the direction
count by one third, but it is still dramatically slower than backprop.

K=512 was not sufficient for A-update fidelity after the second training step.

## Resource and integrity evidence

Runs completed on Xwing's AMD Radeon 8050S using PyTorch
2.12.0+rocm7.14.0. Peak allocated GPU memory was about 2.45 GiB.
Every comparison verified:

- pretrained base `o_proj` weight unchanged;
- no gradients accumulated on base-model parameters;
- zero backward calls in the structured forward-only path;
- no checkpoint or model-weight file written;
- no deployment/scheduler/NAS mutation.

JSON manifests for K=512/1024/1536 were copied without overwrite to Xwing's
local finetune-staging evaluation directory, chmod600, and SHA-256 verified.

## Next gate

K=1024 is now the best candidate for a longer research run. Before using more
fleet GPUs, repeat K=1024 across multiple seeds and 4–8 update steps with a
predeclared learning-rate schedule, then compare:

1. aggregate train CE before/after;
2. heldout CE on a larger independently checked split;
3. adapter A/B update cosine against exact backprop;
4. wall/GPU time and peak memory;
5. agentic/tool-call task outcomes.

The full K=1536 basis should remain an oracle for correctness rather than a
default training configuration. Earlier layers and q/k/v still require
separate future-token causal-credit work.


## Four-step K=1024 accumulation check

A separate seed-42 run expanded to **four updates**, 12 selected training
examples for aggregate CE, and 8 heldout examples. The purpose was to see
whether estimator error compounds after A and B have both moved farther from
initialization.

| Method | train CE delta | heldout CE delta | time |
| --- | ---: | ---: | ---: |
| backprop control | -0.000543 | +0.000315 | 5.49 s |
| K=1024 structured forward-only | **-0.001178** | +0.000635 | 351.67 s |

Final adapter-update alignment versus the exact backprop control:

- LoRA-A cosine: **0.99358**, relative L2 error 0.1134
- LoRA-B cosine: **0.85515**, relative L2 error 0.6047

The A update remained extremely well aligned after four real updates; B
alignment remained positive but less exact. The forward-only run reduced
aggregate training CE more than the simple backprop control in this one seed,
but **heldout CE worsened more**, so this is not evidence of better
generalization. It does show that K=1024 structured forward-only updates stay
directionally coherent across more than the minimum two-step test.

Before extending to 8+ updates, repeat this K=1024 configuration across
multiple seeds and increase independent heldout/task coverage. Do not tune
learning rate against the current eight-example heldout set.
