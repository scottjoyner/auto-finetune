# K2-Horizon cached-tail forward-only training

Research tracking: issue #1. This slice stacks on the K=1024 batching work and
remains isolated from the production trainer, scheduler, serving, checkpoint
promotion, routing, notifications, and NAS recovery.

## Key idea

The current Dust-inspired adapter is attached only to the **final decoder
layer's attention `o_proj`**. Everything upstream of that projection is
frozen and independent of the temporary LoRA factors and output perturbations.

Therefore the expensive prefix can be evaluated once per example and cached:

```
tokens
  -> layers 0..26
  -> final-layer attention through o_proj
  -> CACHE(layer input, o_proj input, clean o_proj output)
  -> add LoRA / perturbation
  -> final residual + MLP
  -> final norm
  -> LM head
  -> supervised next-token CE
```

For training, only supervised next-token positions need the downstream
MLP/norm/LM-head replay. The implementation therefore has two optimizations:

1. **prefix caching**: do not rerun the first 27 layers or final attention for
   each perturbation;
2. **scored-position replay**: do not run the final tail or LM head for masked
   prompt positions that contribute no supervised CE.

This optimization is valid only for the final `o_proj` experiment. Earlier
layers and q/k/v remain out of scope because their perturbations have
downstream causal effects through attention.

## Algebra / implementation acceptance

`experiments/dust/k2_tail_replay.py` captures the frozen final-layer prefix
with forward hooks and replays the exact K2 decoder algebra after `o_proj`.
Temporary rank-4 A/B factors are updated directly from the forward-only
activation-gradient estimator. No pretrained parameter gets a gradient or is
modified.

On the real pretrained K2-Horizon-0.9B model on Xwing ROCm, the formal
benchmark reports **exact zero difference** for batch-1:

- frozen base logits: max absolute difference **0.0**, CE delta **0.0**;
- nonzero LoRA logits: max absolute difference **0.0**, CE delta **0.0**;
- LoRA + activation jitter logits: max absolute difference **0.0**, CE delta
  **0.0**.

Synthetic tests also cover a rectangular attention projection, full-vs-tail
LoRA/jitter equivalence, two-step adapter-update fidelity, and supervised-only
tail scoring.

## Why cached tail is also more faithful than full-model microbatching

Full-model perturbation microbatching repeats identical token sequences in a
larger BF16 batch. On ROCm this changes upstream matrix arithmetic slightly,
so its supposedly identical frozen prefix is not bit-identical to the
single-example serial reference.

Caching the prefix from the single example avoids that source of estimator
drift. At K=256, seed 42, sigma=.25:

| one update | time | B cosine vs serial full-model estimator |
| --- | ---: | ---: |
| serial full model | 17.663 s | reference |
| full-model D=4 batching | 7.350 s | 0.96590 |
| cached full-sequence tail D=4 | 0.720 s | 0.99952 |
| **cached scored-position tail D=4** | **0.470 s** | **0.99957** |

After **two updates**, where both LoRA A and B can move:

| method | time | A cosine vs serial | B cosine vs serial |
| --- | ---: | ---: | ---: |
| serial full model | 38.118 s | reference | reference |
| full-model D=4 | 18.632 s | 0.82591 | 0.96035 |
| **scored-tail D=4** | **1.596 s** | **0.99784** | **0.99941** |

The two-step scored-tail path is **23.89x faster than the serial estimator**
and **11.68x faster than full-model D=4**, while being *closer* to serial.

## Formal K=1024 two-step acceptance

A separate immutable JSON benchmark uses the real K2-Horizon base, two
deterministically selected training examples, K=1024, sigma=.25, rank4,
lr=.1 and D=4.

| metric | full serial reference | cached scored-tail |
| --- | ---: | ---: |
| perturbation forwards | 4098 | 514 tail forwards |
| update time | 154.240 s | **6.358 s** |
| prefix cache | n/a | 0.804 s |
| A-update cosine | reference | **0.999665** |
| B-update cosine | reference | **0.999577** |
| peak allocated GPU memory | — | ~2.61 GiB |

Speedup is **24.26x** for the update loop and **21.54x including prefix
caching**. This is the strongest compute/fidelity result in the experiment so
far.

Increasing tail direction batch from D=4 to D=16 did not improve two-step
wall time on this workload (6.95 s vs 7.04 s) and increased peak allocation.
D=4 remains the preferred setting.

## Four-step, three-seed K=1024 training/evaluation

The optimized path was then run against the exact fixed dataset manifests used
by the earlier serial K=1024 study:

- seeds: 7, 42, 1337;
- 16 train examples / 864 scored assistant tokens;
- 12 prompt-disjoint heldout examples / 624 scored tokens;
- 4 updates;
- K=1024 orthogonal antithetic directions;
- sigma=.25, lr=.1, D=4;
- identical source and selected-pair SHA-256 digests.

Negative CE delta is improvement.

| seed | tail train delta | tail heldout delta | total tail time | serial structured time | backprop time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -0.000011 | +0.000813 | 14.81 s | 346.89 s | 7.46 s |
| 42 | -0.000858 | +0.001127 | 15.31 s | 347.57 s | 6.87 s |
| 1337 | +0.000407 | **-0.000177** | 14.87 s | 347.80 s | 6.86 s |

Aggregate:

| metric | cached scored-tail | serial structured | backprop |
| --- | ---: | ---: | ---: |
| mean train CE delta | **-0.000154** | -0.000149 | -0.000210 |
| mean heldout CE delta | +0.000588 | +0.000580 | +0.000147 |
| train-improved seeds | 2/3 | 1/3 | 2/3 |
| heldout-improved seeds | 1/3 | 0/3 | 0/3 |
| mean total elapsed | **14.99 s** | 347.42 s | 7.06 s |

The optimized forward-only path is **23.17x faster than the prior serial
structured implementation** and is now only about **2.12x the elapsed time of
the matched adapter-only backprop control** for this tiny protocol.

The aggregate train and heldout deltas remain extremely close to the serial
structured reference, which is what should happen when an optimization
preserves the estimator. One seed happens to improve the tiny heldout set, but
the average heldout delta remains positive. This is **not a generalization
win**.

## Resource and custody boundaries

Every live run:

- used the pinned pretrained K2-Horizon-0.9B weight SHA;
- kept the base model frozen and produced no base gradients;
- made zero backward calls in the forward-only path;
- wrote no adapter/model checkpoint;
- respected host-memory admission and in-step safety floors;
- left co-resident inference services running;
- did not touch NAS recovery/migration state.

Per-seed manifests, the aggregate summary, and direct K=256/K=1024 benchmark
manifests are stored on Xwing local SSD in the finetune-staging evaluation
reports directory with mode 600 and recorded SHA-256 digests.

## Interpretation and next gate

The major efficiency bottleneck from the first Dust experiments has changed.
For the constrained final-`o_proj` adapter, forward-only training is no
longer tens of times slower than backprop: the measured matched protocol is
now roughly **2x** slower end-to-end.

That makes the next question about **learning quality**, not basic execution
cost. The next gated experiment should increase useful optimization signal
without widening model authority:

1. keep the cached scored-tail K=1024/D=4 implementation fixed;
2. use a larger independently checked heldout/task suite;
3. test 8–32 updates with a predeclared learning-rate schedule and multiple
   seeds;
4. report heldout CE, tool-call/agentic outcomes, elapsed/GPU time and update
   stability;
5. do not promote a checkpoint unless quality improves repeatably.

Earlier-layer and q/k/v Dust experiments still require separate future-token
credit assignment and should not inherit this tail-cache shortcut.
