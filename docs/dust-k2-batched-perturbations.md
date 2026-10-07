# K2-Horizon perturbation microbatching

This slice addresses the dominant cost discovered in the K=1024 multiseed
study: the structured forward-only path evaluates each +direction and
-direction as separate model forwards.

## Implementation

`structured_step_batched` groups several orthogonal directions in the model
batch dimension and evaluates both signs in the same K2 forward:

- a chunk of D orthogonal directions becomes a 2D model batch;
- the first D rows receive +sigma perturbations and the next D receive -sigma;
- tokenwise antithetic CE differences are reduced back into the exact same
  activation-gradient estimator;
- the clean final-`o_proj` input cache is restored before applying the local
  LoRA update;
- batch sizes are explicitly bounded to 1/2/4/8/16;
- the existing host-memory floor remains active;
- no base-model gradient, optimizer, checkpoint, scheduler or deployment path
  is introduced.

The prior serial estimator remains unchanged and is used as the numerical
reference.

## Synthetic acceptance

On Xwing's ROCm Python environment, the structured training tests now include a
two-step full-basis comparison showing the batched implementation produces the
same tiny-model LoRA result as the serial estimator within floating-point
tolerance. **2 tests passed**.

## Real K2-Horizon K=256 benchmark

Pretrained K2-Horizon-0.9B, seed 42, sigma=.25, one structured update:

| method | model forwards | elapsed | B-update cosine vs serial | B norm ratio |
| --- | ---: | ---: | ---: | ---: |
| serial + / - | 513 | 23.32 s | reference | reference |
| batched D=1 | 257 | 14.69 s | 0.96585 | 1.0175 |
| batched D=2 | 129 | 11.78 s | 0.96947 | 1.0336 |
| batched D=4 | **65** | **9.51 s** | **0.96960** | 1.0291 |

D=4 is approximately **2.45x faster** than the serial implementation in this
bounded K=256 run while retaining a strongly aligned B update.

The cosine is not exactly 1.0 even for D=1 because K2's BF16/ROCm execution is
not bit-identical when the same examples are evaluated in a larger batch. The
small norm-ratio shift and ~0.97 cosine must therefore be included in any
speed/fidelity tradeoff rather than assuming batching is algebraically free.

Peak allocated device memory across the benchmark was about **2.49 GiB**.

## K=1024 gate

A K=1024 D=4/D=8 benchmark was **not launched** when Xwing host-memory
headroom fell below the predeclared >=7 GiB admission threshold. This is a
successful safety gate, not a failed experiment. Current co-resident services
were left untouched.

The next accepted K=1024 benchmark should start only after host memory recovers
above the gate, then compare D=4 and D=8 for:

1. wall time and forward-call count;
2. B-update cosine/norm/error versus the serial K=1024 estimator;
3. peak device allocation and minimum host-memory headroom;
4. whether the resulting speedup survives a two-to-four-step training run.

No production checkpoint or larger training matrix should use batching until
that real K=1024 acceptance passes.
