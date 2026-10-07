# K2-Horizon Dust-inspired gradient calibration

Tracks research issue #1 and stacks on draft PR #5. This slice does **not**
train or save an adapter. It measures how well the activation-noise estimator
used by the Dust-inspired last-layer experiment recovers the exact local
gradient on the real pretrained K2-Horizon-0.9B model.

## Why this gate exists

The matched three-seed pilot in PR #5 used a **one-sided Gaussian estimator,
population 64, sigma 0.05** and worsened held-out CE in all three Dust runs.
That result was valid, but it did not separate an optimization failure from a
high-variance gradient-estimation failure.

For the final decoder layer's attention `o_proj`, downstream computation is
position-local. We can therefore cut the graph at that projection output,
compute exact local `d(mean masked next-token CE)/d(o_proj output)` using
`torch.autograd.grad`, and compare it to forward-only finite-difference
estimators without changing the pretrained base model.

## Implementation

`experiments/dust/k2_gradient_calibration.py`:

- loads the pinned local K2-Horizon-0.9B BF16 weights on Xwing ROCm;
- selects one training pair deterministically by SHA-256 rank without emitting
  prompt, response or token IDs into the report;
- freezes all pretrained parameters and captures the final `o_proj` input;
- computes an exact **local** output gradient by replacing only the final
  projection output with a detached leaf tensor;
- estimates that gradient with **one-sided** and **antithetic** activation
  perturbations;
- maps exact and estimated output gradients into the same rank-4 LoRA-B
  gradient and reports cosine, relative L2 error and norm ratio;
- supports ordinary Gaussian directions and randomized **orthogonal**
  directions scaled to preserve the usual Gaussian second moment;
- writes JSON evidence only when an exclusive new output path is requested.

No optimizer, checkpoint, scheduler, NAS writer, deployment path or production
authority is touched.

## Real Xwing results

Hardware/runtime: AMD Radeon 8050S, PyTorch 2.12.0+rocm7.14.0, pretrained
K2-Horizon-0.9B, final-layer `o_proj`, one deterministically selected
training example with 34 scored assistant tokens.

### Baseline estimator explains the negative pilot

At **Gaussian, one-sided, sigma 0.05, K=64**, the measured rank-4 LoRA-B
gradient cosine was only about **0.106** on the calibration example. In other
words, the estimator configuration used by the first real matched Dust pilot
was a very noisy approximation of the exact local update.

Antithetic sampling and larger perturbations improved alignment materially.

### Gaussian antithetic sweep

For seed 42:

| sigma | population | output-gradient cosine | LoRA-B cosine |
| ---: | ---: | ---: | ---: |
| 0.025 | 256 | 0.2181 | 0.1948 |
| 0.05 | 256 | 0.2998 | 0.2668 |
| 0.10 | 256 | 0.3529 | 0.3365 |
| 0.15 | 256 | 0.3663 | 0.3431 |
| 0.20 | 256 | 0.3718 | 0.3594 |
| 0.25 | 256 | 0.3741 | 0.3614 |

Within the tested range, sigma 0.20–0.25 was better than the original 0.05.

### Orthogonal antithetic directions

Randomized orthogonal directions further reduced variance at the same
population. At sigma 0.25:

| population | output-gradient cosine | LoRA-B cosine | LoRA-B relative L2 error |
| ---: | ---: | ---: | ---: |
| 256 | 0.40162 | 0.40411 | 2.2518 |
| 512 | 0.56927 | 0.54898 | 1.4131 |
| 1024 | 0.79906 | 0.79675 | 0.7435 |
| 1536 | **0.98563** | **0.98125** | **0.1947** |

The 1536-direction run is a complete randomized orthogonal basis for the
1536-wide K2 hidden state. It required **3074 model forwards**, completed in
**121.88 seconds**, and peaked at approximately **2.25 GiB** allocated device
memory.

This is the strongest result so far: with enough well-structured forward
evaluations, the forward-only estimator recovers essentially the same local
LoRA-B update direction as exact autograd.

## Interpretation

This changes the diagnosis of PR #5. The first real Dust pilot should **not**
be read as evidence that the basic forward-only estimator is incapable of
learning K2-Horizon. Its K=64 one-sided Gaussian configuration was simply too
noisy for this 1536-dimensional activation.

The result also does **not** make full-basis Dust economical. A near-exact
1536-direction antithetic estimator needs roughly two forward passes per
direction, making it far more expensive than one backward pass. The useful
engineering question is now the quality/compute frontier between roughly
K=512 and K=1024, not whether the estimator math works at all.

## Next gate

Implement an opt-in training variant using **antithetic orthogonal directions**
at fixed, predeclared populations. First compare K=512 and K=1024 against the
same exact adapter-only backprop control on identical train order and held-out
data. Record train CE before/after, held-out CE, adapter-update cosine, GPU
time and memory. Use the full K=1536 basis only as a calibration oracle, not a
default trainer.

No checkpoint promotion or production integration is justified until a
multi-seed held-out/task evaluation demonstrates a repeatable benefit.
