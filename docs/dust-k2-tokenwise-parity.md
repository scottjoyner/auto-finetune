# K2-Horizon tokenwise perturbation parity

This slice implements the highest-priority follow-up from the Dust/K2 paper
plan: compare our existing shared-direction orthogonal estimator with
**independent per-token Gaussian perturbations** closer to Dust's
token-as-virtual-population semantics.

The experiment remains deliberately narrow:

- frozen pretrained K2-Horizon-0.9B;
- final decoder-layer attention `o_proj` only;
- rank-4 LoRA;
- cached scored-tail replay;
- exact local autograd used only as a calibration reference;
- no checkpoint, scheduler, deployment, NAS recovery, routing, or serving
  changes.

## What changed

`k2_tail_replay.py` now exposes two forward-only scored-tail estimators:

1. `shared_orthogonal`
   - one orthogonal direction per draw;
   - the same direction is used at every scored token position;
   - directions are antithetic and randomized.

2. `tokenwise_gaussian`
   - every scored token receives an independent Gaussian direction for every
     draw;
   - positive and negative perturbations are evaluated antithetically;
   - this is closer to Dust's independent per-token activation-noise
     semantics, but it is still not a reproduction of Dust's full
     earlier-layer/attention-credit algorithm.

The training comparison harness can now select either estimator with
`--estimator shared_orthogonal` or
`--estimator tokenwise_gaussian`.

## Exact-gradient calibration protocol

The real pretrained K2-Horizon-0.9B model was evaluated on Xwing ROCm with:

- 34 scored assistant-token positions;
- final-`o_proj` hidden width 1536;
- rank-4 LoRA;
- a reproducible small nonzero LoRA-B calibration state so both A and B
  gradients are measurable;
- estimator seeds 7, 42, 1337;
- populations K=64, 256, 1024;
- sigma 0.05 and 0.25;
- D=4 perturbation microbatching.

For every configuration we compare:

- local output-gradient cosine to exact local autograd;
- LoRA-A gradient cosine;
- LoRA-B gradient cosine;
- population/runtime/memory telemetry.

The two estimators use the same number of perturbation draws and
token-position perturbations. The distinction is the number of **unique
direction vectors**:

- shared orthogonal: K unique vectors reused across the 34 positions;
- tokenwise Gaussian: K x 34 independently sampled vectors.

## Three-seed result

Mean cosine to exact local autograd:

| estimator | sigma | K | output cosine | LoRA-A cosine | LoRA-B cosine | mean time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared orthogonal | .05 | 64 | .1518 | .0173 | .1544 | .123 s |
| tokenwise Gaussian | .05 | 64 | .1546 | .1256 | .1713 | .123 s |
| shared orthogonal | .05 | 256 | **.3106** | **.1998** | **.3389** | .594 s |
| tokenwise Gaussian | .05 | 256 | .2966 | .1001 | .3254 | .483 s |
| shared orthogonal | .05 | 1024 | **.6314** | **.6541** | **.6754** | 1.898 s |
| tokenwise Gaussian | .05 | 1024 | .5242 | .4228 | .5592 | 1.869 s |
| shared orthogonal | .25 | 64 | .1964 | .1984 | .1899 | .117 s |
| tokenwise Gaussian | .25 | 64 | .1976 | .2059 | .2033 | .116 s |
| shared orthogonal | .25 | 256 | **.4002** | **.3401** | **.4075** | .467 s |
| tokenwise Gaussian | .25 | 256 | .3716 | .2344 | .3780 | .466 s |
| shared orthogonal | .25 | 1024 | **.8091** | **.7278** | **.8116** | 1.875 s |
| tokenwise Gaussian | .25 | 1024 | .6266 | .5024 | .6274 | 1.868 s |

At K=64 the estimators are broadly comparable and tokenwise Gaussian has a
small mean advantage in several metrics. That advantage does **not** survive
larger populations.

At K=256 and K=1024, the existing shared orthogonal estimator is consistently
better aligned with exact local gradients at essentially the same measured
tail runtime. At sigma=.25 and K=1024:

- shared orthogonal: output/A/B cosine = **.8091 / .7278 / .8116**;
- tokenwise Gaussian: **.6266 / .5024 / .6274**.

The tokenwise A-gradient estimate also shows substantially larger seed
variance at K=1024 (population standard deviation ~.2066 vs ~.0482 for the
shared orthogonal control).

## Interpretation

This closes an important semantic gap with Dust, but it is a **no-go for
replacing the current K=1024 estimator**.

Independent per-token Gaussian noise works: it produces a finite,
directionally meaningful estimate and improves as K grows. However, in this
special final-`o_proj`, position-local setting, independent token noise does
not provide a quality advantage at equal draw count. Each scored token's loss
depends only on its own final-tail perturbation, so reusing a well-structured
orthogonal direction basis across positions does not create cross-token
credit contamination. Orthogonality instead appears to reduce variance.

This should not be read as a contradiction of Dust's pretraining result.
Dust's tokenwise mechanism matters in a broader architecture where many
linear sites are perturbed and attention creates cross-token/future-token
credit. Our current tail experiment deliberately removes most of that
structure.

The useful result is narrower:

> For final-layer position-local adapter tuning, independent per-token
> Gaussian activation noise is viable but less sample-efficient than shared
> orthogonal antithetic directions at K=256--1024.

That distinction should be included in the paper rather than claiming parity
where the algorithmic setting still differs.

## End-to-end tokenwise training smoke

The tokenwise estimator was also wired into the cached-tail training harness
and executed for two real updates at K=256, sigma=.25, seed 42:

- train CE delta: **-0.001189**;
- heldout CE delta: **+0.000130**;
- forward-only training time: **1.906 s**;
- total bounded comparison time: **3.845 s**;
- backward calls: **0**;
- frozen base unchanged: **true**.

This proves the selectable tokenwise path is operational end-to-end. The
heldout result is not a quality claim and does not justify scaling that
estimator after the calibration no-go.

## Evidence

Public-safe manifests were copied without overwrite to Xwing local SSD:

- `tokenwise-parity-3seed-k64-k256-k1024-s005-s025.json`
  - SHA-256:
    `09ca48e8ee09953e5aa5f81dd55bdced4d5ca3f0e223c4fe1f4cbd0b05d49943`
- `tokenwise-train-smoke-k256-seed42.json`
  - SHA-256:
    `857ff411a0e621ba16ffce7e8912a5ce4740761876ab411914043d9f7daa33dd`

Both contain aggregate metrics/digests only; no prompts, responses, or token
IDs are serialized.

## Next gate

Keep **shared orthogonal K=1024 / sigma=.25 / D=4** as the fixed estimator for
the next quality slice.

Spend the recovered compute budget on:

1. 8/16/32 update schedules;
2. a larger independently checked heldout split;
3. near-duplicate screening beyond exact normalized prompts;
4. fixed tool-call JSON/exactness evaluation;
5. fixed agentic completion tasks;
6. 3--5 predeclared seeds;
7. matched adapter-only backprop.

Do not revisit estimator selection using heldout quality. The parity gate is
closed: tokenwise Gaussian was tested and did not beat the structured control
at the operating population.
