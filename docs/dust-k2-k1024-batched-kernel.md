# K2-Horizon K=1024 low-overhead batching acceptance

This slice follows draft PR #9. It isolates the perturbation kernel from
heldout scoring so K=1024 batching can be measured without the host-memory
pressure of a full train/eval comparison. It does **not** establish model
quality and does not write an adapter checkpoint.

## Protocol

- real pretrained K2-Horizon-0.9B BF16 on Xwing ROCm;
- final decoder-layer attention `o_proj`, fresh rank-4 LoRA;
- deterministic SHA-256 selected training examples;
- K=1024 orthogonal antithetic perturbations, sigma=0.25, lr=0.1;
- serial estimator remains the numerical reference;
- batched variants put D positive and D negative directions into one model
  batch;
- all pretrained parameters frozen; zero backward calls;
- per-chunk `MemAvailable` telemetry added to both serial and batched paths.

Because this kernel probe omits aggregate train/heldout scoring, it has a
separate **>=5 GiB host-memory start gate** while retaining the existing
1.5 GiB in-step abort floor. This lower start gate applies only to this
one/two-step microbenchmark. The full K=1024 training comparison keeps its
higher >=7.5 GiB external admission threshold.

## One-step K=1024 result

| mode | model forwards | wall time | B cosine vs serial | B relative L2 error | minimum host MemAvailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial | 2049 | 80.10 s | reference | reference | 3.86 GiB |
| D=4 | **257** | **36.62 s** | **0.96701** | 0.2573 | 2.13 GiB |
| D=8 | **129** | **33.45 s** | 0.93332 | 0.3772 | 2.41 GiB |

D=4 is approximately **2.19x faster** than serial while retaining stronger
update fidelity than D=8. D=8 is ~2.39x faster but introduces more drift.

The forward-call reduction is much larger than the wall-time reduction because
larger model batches increase work per forward; the speedup is therefore
bounded by actual GPU throughput rather than Python/model-call overhead alone.

## Two-step K=1024 D=4 result

Two sequential deterministic training examples were used so LoRA-B is nonzero
before the second update and LoRA-A receives a meaningful update.

| mode | model forwards | wall time |
| --- | ---: | ---: |
| serial | 4098 | 164.35 s |
| D=4 | **514** | **83.34 s** |

Speedup: approximately **1.97x**.

Final batched update versus the serial structured estimator:

- LoRA-A cosine: **0.98050**, relative L2 error **0.1985**
- LoRA-B cosine: **0.96099**, relative L2 error **0.2817**
- minimum host MemAvailable during D=4: **2.44 GiB**
- peak allocated device memory: approximately **2.69 GiB**

This is the important acceptance: batching remains strongly aligned after the
second real K2 update, when both LoRA factors can move. D=4 is therefore the
preferred batching point from the tested set.

## Interpretation

The current cost frontier has improved materially:

- serial K=1024: highest fidelity, highest cost;
- D=4 K=1024: roughly half the wall time with ~0.98 A / ~0.96 B update cosine
  after two steps;
- D=8: a little faster for one step, but lower B fidelity (~0.93), so it is
  not the preferred default.

This still does **not** make forward-only training competitive with
adapter-only backprop. The prior four-step serial K=1024 experiment took
hundreds of seconds versus single-digit seconds for backprop. Batching attacks
the largest obvious implementation inefficiency but does not erase the
fundamental extra-forward cost.

## Evidence / custody

The K=256 benchmark, K=1024 one-step D=4/D=8 benchmark, and K=1024 two-step
D=4 benchmark were copied without overwrite to Xwing local SSD under the
finetune-staging evaluation reports directory, chmod600. SHA-256 digests were
recorded. NAS recovery/migration state was not touched.

## Next gate

Wire D=4 into an **opt-in research-only** structured backend and repeat the
existing 4-step, three-seed K=1024 protocol. The acceptance criteria should
remain:

1. A/B update cosine versus serial structured reference;
2. wall-time and GPU-memory reduction;
3. unchanged frozen base and zero backward calls;
4. the same fixed train/heldout manifests;
5. no checkpoint promotion if heldout/task quality remains negative.

The quality problem is still open. This optimization only makes it cheaper to
study.
