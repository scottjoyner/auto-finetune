# Next research slice: closer Dust parity, quality scaling, and a contribution package

## Objective

Build directly on PR #11's strongest result: for a **frozen pretrained
K2-Horizon-0.9B** model with a rank-4 LoRA attached only to the final
decoder-layer attention `o_proj`, cached scored-tail replay preserves the
forward-only estimator while reducing the four-step K=1024 protocol from
~347 s to ~15 s on average, about **23x faster than the prior serial
implementation** and about **2.1x the matched adapter-only backprop time**.

The next slice should spend that recovered compute budget on a more faithful
comparison to Dust and on **quality**, not on wider production authority.

The upstream Dust work perturbs activations independently at every token and
uses token-local loss changes as credit for linear-layer outputs. Our current
K2 experiment is explicitly **Dust-inspired**, not a reproduction: its
orthogonal directions are shared across token positions within each draw.
That distinction is scientifically important and is the first thing this
slice should close.

## Research questions

1. **Tokenwise parity:** does independent per-token activation noise, closer
   to Dust's estimator semantics, improve gradient fidelity or population
   efficiency for pretrained adapter tuning?
2. **Population efficiency:** after cached-tail replay removes most frozen
   prefix cost, what K is actually necessary for stable LoRA updates?
3. **Quality:** can a fixed forward-only estimator produce repeatable heldout
   or task-quality gains over 8--32 updates, rather than merely matching the
   direction of backprop?
4. **Portability:** do the results remain stable on AMD ROCm and, when a
   compatible idle GPU is available, a second accelerator/runtime?
5. **Reproducibility:** can we publish a private-data-safe evidence package
   that another group can run without our local corpus?

## Slice A -- Dust-parity estimator

Implement a new, opt-in `tokenwise_antithetic` estimator in the cached-tail
research path.

### Semantics

For every scored token position `t` and draw `k`, sample an independent
perturbation `u[k,t,:]`. Evaluate antithetic losses

```
L(h + sigma * u[k,t,:])
L(h - sigma * u[k,t,:])
```

and estimate the local output gradient from token-local loss differences.
Because this experiment modifies the **final `o_proj`**, no future-token
attention credit is required after that site: the remaining MLP/norm/LM-head
tail is position-local. Earlier layers and q/k/v remain out of scope.

Keep the current shared-direction orthogonal estimator unchanged as a control.

### Acceptance matrix

Use one deterministic training example first:

| estimator | K / draws | sigma | target |
| --- | ---: | ---: | --- |
| shared Gaussian one-sided | 64 | .05 | historical baseline |
| shared orthogonal antithetic | 256, 512, 1024 | .25 | current control |
| tokenwise Gaussian antithetic | 64, 256, 1024 | tuned only from train-side calibration | Dust-parity candidate |

Report:

- local output-gradient cosine vs exact autograd;
- LoRA-A / LoRA-B gradient cosine and relative L2 error;
- estimator variance across at least 5 seeds;
- number of tail forwards, scored tokens, and effective perturbation samples;
- wall time and peak device / host memory;
- zero backward calls in the forward-only path.

Do **not** tune against heldout loss.

### Go/no-go

Advance a tokenwise estimator only if it either:

- reaches the same gradient cosine at materially lower K / wall time, or
- reaches materially higher cosine at the same compute budget.

A null result is useful and should be retained.

## Slice B -- quality scaling with the fixed best estimator

Once Slice A picks the estimator, freeze it before quality experiments.

### Protocol

Use the same deterministic source selection and prompt-disjointness gates as
PR #11, but expand evaluation before training:

- 32--64 train examples;
- at least 32 independently checked heldout examples;
- exact prompt overlap rejection plus a near-duplicate screen;
- tool-call exactness / JSON validity where applicable;
- a small fixed agentic completion suite;
- 5 seeds if runtime permits, otherwise predeclare 3.

Run **8, 16, and 32 update** schedules. Predeclare one learning-rate schedule
from training-side evidence only. Do not select a checkpoint from heldout.

Compare:

1. frozen base;
2. matched adapter-only backprop;
3. forward-only cached-tail treatment.

Primary outcomes:

- heldout token CE / perplexity;
- tool-call exactness and structural validity;
- fixed agentic-task completion;
- A/B update cosine against the matched backprop trajectory;
- elapsed time and GPU-hours.

Promotion requires repeatable quality improvement. A train-loss decrease or
high gradient cosine alone is insufficient.

## Slice C -- public reproducibility package

Create a public-safe research bundle separate from private training content:

- synthetic fixture that exercises the exact estimator path;
- manifest schema and SHA-256 provenance fields;
- CLI examples with redacted/local paths;
- no prompts, responses, token IDs, secrets, hostnames, or credentials;
- one JSON schema for calibration, training, and quality summaries;
- a results table generated only from immutable manifests.

If upstream Q Labs is contacted, send this public-safe package and the paper
draft, not local corpus material.

## Proposed paper contribution

Frame the work as an **independent systems extension and adapter-fine-tuning
study of Dust-inspired activation-space zeroth-order optimization**, not as a
reproduction of the full Dust pretraining result.

The contribution is strongest when it includes both positive and negative
findings:

- activation perturbation updates on a pretrained 0.9B transformer can align
  closely with adapter-only backprop;
- full orthogonal populations recover near-backprop update directions;
- cached final-tail replay removes most avoidable compute in the constrained
  last-`o_proj` setting;
- the optimized forward-only path is ~2.1x matched backprop in the current
  tiny protocol rather than ~49x;
- **quality has not yet improved on average**, so update fidelity and
  generalization must be distinguished;
- the current estimator is not full Dust because tokenwise independent noise,
  earlier-layer credit, and attention future-credit remain untested.

This is useful evidence for the broader Dust thesis precisely because the
limitations and negative results are explicit.

## Deliverables

1. `tokenwise_antithetic` cached-tail research backend + tests.
2. Gradient-alignment / variance matrix with immutable manifests.
3. Expanded quality protocol and predeclared schedule.
4. Public-safe reproduction fixture and result summarizer.
5. arXiv-compatible draft under `papers/dust-k2-extension/`.
6. A concise upstream note to Q Labs after the parity experiment is complete.

## Authority boundary

This slice remains research-only. No automatic scheduler admission, checkpoint
promotion, serving change, NAS recovery mutation, routing change, or
notification-authority change is part of this plan.
