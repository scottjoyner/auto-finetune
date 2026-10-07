# K2-Horizon K=1024 structured forward-only multiseed acceptance

This slice follows the update-fidelity work in draft PR #7. It asks whether
the current practical candidate, K=1024 orthogonal-antithetic perturbations,
remains directionally coherent across seeds and whether a larger fixed heldout
slice shows any repeatable generalization signal.

## Fixed protocol

All three runs use the same pretrained K2-Horizon-0.9B BF16 base and:

- final decoder-layer attention `o_proj` only;
- fresh rank-4 LoRA, same initialization rule per seed for backprop and
  structured treatments;
- **4 update steps**, K=1024 orthogonal antithetic directions per step;
- sigma=0.25, learning rate=0.1;
- **16 SHA-256-selected training examples / 864 assistant tokens**;
- **12 prompt-disjoint Hermes-reasoning heldout examples / 624 assistant
  tokens**;
- train/eval source and selected-pair SHA-256 digests fixed across seeds;
- seeds 7, 42 and 1337.

The structured path uses zero backward calls. The backprop comparison computes
gradients only for the fresh LoRA A/B factors. Base-model parameters remain
frozen and no checkpoint is written.

## Results

Negative CE delta is improvement.

| Seed | backprop train Δ | backprop heldout Δ | structured train Δ | structured heldout Δ | A-update cosine | B-update cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -0.000262 | +0.000121 | +0.000015 | +0.000655 | 0.98100 | 0.83090 |
| 42 | -0.000380 | +0.000107 | -0.000860 | +0.000896 | 0.99358 | 0.85515 |
| 1337 | +0.000013 | +0.000214 | +0.000397 | +0.000191 | 0.97917 | 0.82289 |

Aggregate:

| Metric | backprop | structured forward-only |
| --- | ---: | ---: |
| mean train CE delta | **-0.000210** | -0.000149 |
| train-improved seeds | 2 / 3 | 1 / 3 |
| mean heldout CE delta | +0.000147 | **+0.000580** |
| heldout-improved seeds | **0 / 3** | **0 / 3** |
| mean runtime | 7.06 s | 347.42 s |

Structured/backprop mean runtime ratio: **49.19x**.

Across seeds, final adapter-update fidelity remains high:

- LoRA-A cosine mean **0.98458**, minimum **0.97917**
- LoRA-B cosine mean **0.83631**, minimum **0.82289**

## Interpretation

This is a strong reproducibility result for **update direction**, and a
negative result for **generalization**.

K=1024 is no longer a one-seed curiosity: the A update stays near the exact
backprop direction in all three seeds, while B remains consistently positive
but less exact. However, neither the structured method nor the simple
backprop control improved the fixed heldout CE in any seed. The structured
method's average heldout degradation is larger.

Therefore this work does **not** justify checkpoint promotion or a larger
training budget solely on quality grounds. The current bottleneck has moved
from estimator correctness to two areas:

1. compute efficiency: the implementation evaluates perturbations serially;
2. training/evaluation design: four last-layer updates on 16 examples are too
   small to establish useful fine-tuning quality.

## Operational finding: shared-memory admission matters

The first seed-1337 attempt correctly stopped when Xwing host
`MemAvailable` fell below the existing 1.5 GiB in-step safety floor while
other inference services remained active. No result file was emitted. After
the experiment process exited, host headroom recovered. A retry admitted only
after **>=7.5 GiB available host memory** and completed normally.

For K=1024 on Xwing's UMA-style AMD GPU, use >=7.5 GiB host-memory headroom as
the external start gate while current co-resident services remain active.
Do not stop or preempt those services just to obtain a research result.

## Evidence

The three per-seed manifests and aggregate summary were copied without
overwrite to Xwing local SSD, chmod600, and SHA-256 verified. The aggregate
summary enforces identical model, hyperparameters, source digests and selected
train/eval pair digests before combining runs. Raw prompts, responses and
token IDs are not serialized.

## Next engineering gate

Before spending ~50x backprop time on longer experiments, batch multiple
orthogonal perturbation directions in a single K2 forward call. Because the
current implementation evaluates one +direction and one -direction serially,
there is substantial opportunity to increase GPU occupancy while preserving
the exact same finite-difference estimator.

Test small perturbation microbatches first, under the same host-memory safety
floor, and verify numerical equivalence to the serial K=1024 estimator before
repeating multi-seed quality runs.
