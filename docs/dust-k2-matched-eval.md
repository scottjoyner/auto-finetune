# K2-Horizon real-weight Dust-inspired vs backprop-LoRA: matched pilot

Research tracking: issue #1. Stacked on **draft PR #4**. This is an
isolated, opt-in research harness; the automatic fine-tuning scheduler,
model-serving endpoints, notification authority, and NAS migration remain
untouched.

## Scientific question and scope

Can a rank-4 LoRA on **the final decoder layer's** `self_attn.o_proj`
receive useful updates from Gaussian activation-noise estimates of
masked, next-token cross-entropy on K2-Horizon-0.9B?

`experiments/dust/k2_matched_compare.py` compares **two fresh LoRAs**
attached to the *same pretrained BF16 base weights*, with the *same*
initial LoRA seeds and train-order, supervised on the same cleaned
local corpus and evaluated using the same held-out sequences:

- **Backprop control:** exact `torch.autograd.grad` for adapter A/B only;
  all base-model parameters frozen. Plain SGD, no optimizer history.
- **Dust-inspired treatment:** one clean forward plus K independent
  Gaussian-perturbed forwards, difference in tokenwise causal CE,
  direct reward-weighted output-noise estimate, local `o_proj`
  low-rank factor updates; **no backward/autograd calls**.
- **Frozen base:** held-out CE before either adapter is attached.

This is not a reproduction of the complete Q Labs Dust architecture,
population allocation, attention-credit estimator or scaling result.
It does not cover earlier-layer attention projections or q/k/v credit.

## Dataset integrity: important live finding

The already-created `held-out-combined` file is **not a safe small
held-out set** for this subset of `train.combined`: all 6 pairs with
<=256 total tokens in that initial selection shared a normalized
user prompt with the train source. That did not prove all examples
were leaked; it did reveal exact prompt overlap in the usable short
sample, which correctly failed admission.

The explicitly selected `held-out-hermes-reasoning` split provides
separate short-ish pairs at <=512 tokens after rejecting every candidate
whose normalized user prompt appears anywhere in the selected training
source. The data gate performs a deterministic SHA-256-ranked pick,
not cherry-picking examples based on results. It checks exact normalized
prompts, not semantic or MinHash near-duplicates.

The program **does not write selected prompts, responses, token IDs,
or extracted training records** anywhere. Every source is scanned
in-place and the run manifest stores source digests, selected
pair hashes, counts, token budgets, and aggregate losses only.
Source material may contain private agent conversations; do not add
it to a public PR or pass it to external provider APIs.

## Admitted hardware / resource boundaries

GPU acceptance independently confirmed the actual pretrained
K2-Horizon-0.9B **loads and forward-passes on Xwing ROCm 7.14**,
AMD Radeon 8050S, PyTorch 2.12.0+rocm7.14.0.
For the 8-token load probe, peak allocated device memory was
about **2.09 GiB** and the output logits were finite; this is *not*
a memory bound for longer, backward-capable sequences.

Live comparisons are limited to at most **12 update steps**, **64 draws
per Dust step**, **32 training / 12 held-out examples**, and
**480 seconds per backend**. The initial GPU admission requires
6 GiB available device address space and 5 GiB available system RAM.
Active-training conflicts are checked outside the runner using
`flock` and explicit process checks; the run does not preempt
existing inference workloads. Real training artefacts are never
saved or merged in this slice.


## Explicit execution, on authorized Xwing only

~~~bash
# This module uses the existing K2 + torch ROCm virtualenv and local files.
# External flock/timeout protects against concurrent experiment instances.
PYTHONPATH=/path/to/auto-finetune \
flock -n /tmp/k2-dust-matched.lock \
timeout --signal=TERM --kill-after=6 375 \
  /media/scott/data/finetune-venv/bin/python \
  /path/to/auto-finetune/experiments/dust/k2_matched_compare.py \
  --model-dir /path/to/local/K2-Horizon-0.9B \
  --train-jsonl /path/to/existing/clean/train.combined.jsonl \
  --heldout-jsonl /path/to/existing/held-out-hermes-reasoning.jsonl \
  --device cuda --seed 42 --train-count 8 --eval-count 6 \
  --train-max-tokens 128 --eval-max-tokens 512 \
  --steps 8 --draws 32 --sigma 0.05 --lr 0.1 \
  --max-seconds 170 --output /path/to/new/unique-manifest.json
~~~

Do not accept a headline loss improvement as a proof of model quality.
The held-out sample is too small for such a claim; tool-call/agentic
success and anti-memorization benchmarks are not included.

## Acceptance and next gates

1. Tests demonstrate causal masking, exclusive-create evidence, exact
   prompt-overlap rejection and unchanged original base-model weights.
2. Backprop and Dust initialize identical final-layer rank-4 A/B factors.
   Each reports loss, wall time, GPU allocation, number of draws, and
   the fraction of already-frozen targets updated.
3. A three-seed series, with a separate fixed held-out set and
   **negative results archived**, must precede expanded budgets.
4. Before scaling beyond the last o_proj, implement future-token causal
   attention credit. Before publishing a checkpoint, prove NAS custody,
   runtime deployment rollback, and authorization boundaries.
5. Any evidence file saved to NAS must first pass a mount/source/UUID
   check. No local source weights may be removed to clear space.


## Executed comparison — Xwing, October 6–7, 2026

The first paired real-weight smoke ran with 4 train examples, 2 heldout
examples, 1 update, 4 perturbations, seed 42. It passed all model/adapter
integrity gates, but the heldout loss was **worse** for both methods:
backprop `+0.000178` versus Dust `+0.004083` CE. The run therefore
did **not** establish a successful fine-tune.

Next, a fixed three-seed pilot completed with **12 train examples**,
**8 heldout examples (405 supervised tokens)**, **12 update steps**,
**64 perturbations per Dust update**, rank 4, learning rate 0.1,
sigma 0.05, and the same SHA-256 selected train/heldout pairs in each
run. The baseline heldout CE was **2.316777** in every run.

| Seed | Backprop heldout CE delta | Dust heldout CE delta | Backprop train time | Dust train time |
| --- | ---: | ---: | ---: | ---: |
| 7 | +0.001028 | +0.000974 | 2.93 s | 31.56 s |
| 42 | -0.000213 | +0.001524 | 3.17 s | 31.64 s |
| 1337 | +0.000714 | +0.001612 | 2.90 s | 31.72 s |
| **Mean** | **+0.000509** | **+0.001370** | **3.00 s** | **31.64 s** |

**Smaller is better. Neither training method provides a convincing
heldout improvement** on this extremely small set. Dust worsened
heldout CE for all three seeds and was approximately **10.55x**
slower in the method-timed loop. This is a useful **negative pilot
result**, not evidence that Dust is globally ineffective or that
backprop is successful. Do not increase the deployment budget based
on these outcomes.

The maximum reported device memory allocated was ~**2.46 GiB**.
Every run reported unmodified pretrained base weights, absent base
gradients, **zero backward calls for Dust**, and no checkpoint writes.
The library emitted `rocSHMEM/libnuma`, experimental AMD attention,
and missing model docstring warnings, but every bounded comparison
completed. No serving, scheduling or NAS mutation was performed.

Immutable JSON result copies were placed on **Xwing's local SSD**
under its finetune-staging evaluation reports directory with
read-protected permissions. Each copy was verified against its
source using SHA-256; the NAS migration/recovery boundary remains
untouched.

### What to do next (without widening authority)

Check clean **train CE before/after** (not just per-step losses from
different examples) and estimate the cosine/variance of local
activation-perturbation gradients against the analytic last-layer
backprop reference. Sweep perturbation population and learning rate
within a predeclared GPU-minute budget, while retaining the same
heldout data for reporting only. A larger heldout set and independent
task-completion probes are necessary before deciding the method's
value. Do not tune against this eight-example pilot holdout.
