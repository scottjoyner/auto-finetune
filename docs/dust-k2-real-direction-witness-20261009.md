# E3 — real K2 antithetic direction observation + independent x1 receipts

**2026-10-09 | HIGH PRIORITY | Draft research only | Source PR #22 / Dust/K2 PR #13 | NO trainer authority**

## Executive result

The pretrained **K2-Horizon-0.9B** cached final-`o_proj` antithetic estimator now has an **opt-in, observation-only pre/post recorder**. The unchanged shared-orthogonal estimator still scores all K directions. Before each batch's forward evaluation, the producer (Xwing) fsyncs scalar-only PRE evidence and waits for an HMAC-authenticated acknowledgment from the independently owned x1-370 receiver. After the original perturbed forward completes, the producer fsyncs per-direction POST events and a strict derived-label record. The receiver owns its HMAC key and can independently join the submitted source ledger with its signed receipts and the derived labels.

**Observed on actual K2 ROCm weights:** 64/64 real directions recorded with **16/16 x1 PRE receipts**, and **64/64 derived labels reconciled by x1 against their PRE and POST losses**. The K2 base checkpoint remained frozen and the LoRA A/B arrays remained unchanged. **No training step, backward call, checkpoint output, GPU model deployment, hosted provider or NAS write.** This is observation and data-contract validation, not a fitted predictive classifier.

## Exact code and authority boundary

- `k2_tail_replay.py` adds optional `probe_observer=None` to `tail_scored_structured_estimate`; without an observer it follows the existing estimator path. For every original batch it emits pre/post callbacks without changing the orthogonal directions, antithetic scores, population or estimator formula. It is NOT in `tail_scored_structured_step`'s optimizer path by default.
- `k2_direction_witness.py` creates mode-600, exclusive-create scalar event and label files. PRE events include only eight per-direction *pre-probe proxy* scalars, keyed prompt-group digest, sigma, clean loss and local monotonic time. No raw prompts, token IDs, logits, hidden tensors, directions, target text, secrets or model weights are written. Each PRE is flushed/fsynced before the perturbed forward. The receiver acknowledgment is recorded in the same producer hash chain before POST.
- `k2_receipt_receiver.py`: independent x1 receiver stores HMAC-signed `run_id`, PRE-batch hash, receiver-owned timestamp and chained receipt MAC, mode 600. Its 32-byte signing key was created and retained *only* on x1 in a mode-700 research folder; it never crosses to Xwing. Receiver verifies MAC chain, duplicate requests, PRE/batch/POST order and exported labels. x1 is a **different custody process/node**; it is not an HSM or independent ground-truth verifier for inference numerical correctness.
- `k2_real_direction_probe.py`: opt-in `--observe-only` local K2 model/dataset (no download), required exact weight SHA, read-only tokenization, fixed `sigma=.25`, shared orthogonal population 8/16/32/64, direction batch 4, seed 7/42/1337. Verifies no LoRA/base mutation or base gradients. No optimizer step, no promotion and no default scheduled action. A bounded model operation uses cached prefix and only scored final-tail positions.
- `predictive_probe_contract.py` from PR #22 strictly validates complete scalar-only label records; it **always returns `classification_training_authorized=false`** until independent source custody, data adequacy and training-split screening are accepted.

### Important prompt-group correction before acceptance

The initial smoke and two preliminary 64-direction runs pseudonymized **full token and target content**. This could split different answers to the same question into different groups. Their source split assignments are scientifically unsafe and **MUST NOT** enter classifier train/validation/test. They remain **producer/custody smoke evidence only**.

The final accepted source-grouping code, `masked-prompt-prefix-v2`, computes a keyed HMAC from only the **non-supervised, masked prompt prefix**. It rejects samples without a masked prompt prefix. Tests verify that two different target responses to the same prompt map to the same episode group. The original prompt tokens are never serialized. **Near-duplicate prompts are still not screened and constitute a separate HOLD.** No user-provided prompt content is included here.

## Real observed Xwing outcomes

**Host:** xwing, existing ROCm AMD Radeon 8050S environment, existing local K2-Horizon checkpoint, Torch 2.12.0+rocm7.14.0. **Expected pinned pretrained weights SHA256:** `6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365`. Dataset source: preexisting local train selection, source text excluded from exported report. Received no model downloads.

**Final accepted v2 cohort:** 64 directions, 53 scored positions, sigma 0.25, direction batch 4, fixed seed 42, sample source index 1. The observed perturbation/receipt phase took **6.629 seconds**. Positive plus-direction mean CE gains: **42/64**. This is classification-label prevalence, not held-out next-token quality improvement. All 64 were scored (no selective missing-label issue). The source group's hash-derived partition was **validation: 64 direction rows in ONE episode**, not 64 independent examples. No classifier fitted on these labels.

**Final cross-node HMAC audit:** x1 independently verified:
- **16 receiver HMAC receipts**, each carrying SHA of a fsynced PRE batch;
- **64 complete PRE/POST direction pairs**, **144** producer events (64 PRE, 16 RECEIPT, 64 POST);
- hash-chain continuity, per-batch digest/receiver MAC equality, posted score ordering;
- **64/64 derived label fields** against PRE features and clean CE plus POST antithetic CE, pinned model revision, index and grouping; **42** positive directions;
- no source HMAC or receiver signing key disclosure.

Immutable **accepted v2** SHA256 evidence:
- Xwing PRE/POST: `bbf3852d94c7c2f65d2ce544adadcc57980af19e25325a51bd1ba944c6d8f51a`;
- Xwing derived labels: `c3b39beede678b5eeb2382e79798e74ab0abec8785de6769029fd7ece54e2bf3`;
- Xwing summary: `01ec451314e165b399f803260c28ab38eeaca5f82ce9427b38cf5be27557155b`.
- x1 receiver audit: verified at HMAC-receipt run ID `223344556677889900aabbccddeeff00`; full event and label hash joins PASS. x1 signing key is *not* in repo or conversation.

All raw evidence remains on **local SSD**: Xwing `/media/scott/data/finetune-staging/research-witness-20261009/` (mode 700 and mode-600 files); x1 `/home/scott/git/dust-k2-receiver-custody-20261009/` (mode 700 + mode-600 key/receipts/audit copies). Neither is on NAS5. Do not upload real derived per-direction data to GitHub or expose it through hosted CI.

## Tests, explicit failures, independent custody

- Five K2 synthetic estimator/witness tests on x1 CPU Torch and **three receiver-only stdlib tests**: PASS at branch `1d4dbfc` or later. The live Xwing runner also completed the 64-direction protocol. Receipt/label audit was independently executed on x1.
- Focused GitHub workflow runs receiver HMAC negative tests, source-label privacy contract tests and `py_compile`; it does **not** download K2 or substitute for actual ROCm model-parity acceptance.
- A transient local test failed due to using `fdopen(...).name` (an integer descriptor) as a filename. It was fixed to preserve the explicit path before live K2 use.
- An incorrect full-content HMAC group definition was discovered after preliminary collection, then fixed and explicitly superseded before the accepted v2 run.
- All tests and models ran in isolated Git worktrees; production worktrees, schedulers, NAS recovery services and the ordinary Dust optimizer remained untouched.

## Remaining gates — no premature classifier training

1. **More independent source episodes:** at least 32 true prompt-disjoint heldout *episodes* and adequate train/validation source groups, plus prompt/near-duplicate screening. One v2 validation episode is nowhere near enough. Do not turn 64 directions from one episode into 64 independent examples.
2. **Classifier feature quality:** this first eight-scalar real-probe profile uses cached activation geometry and zero placeholders for unavailable historical momentum signals. It is **not the same feature distribution as the toy quadratic classifier** in PR #22. Freeze a genuine real feature schema and collect history only from previously observed episodes before fitting; the toy classifier cannot be transferred to these columns.
3. **Independent custody claim precision:** x1 independently attests PRE-batch commitment and hashes; it cannot attest true wall-clock consistency or ground-truth numerical forward losses. The receiver verifies exported labels against the source event ledger but cannot independently recompute K2 inference.
4. **No selective logging:** any future classifier must be evaluated with complete K direction counterfactuals under the same compute budget before using learned selection to change the perturbation distribution. Importance weighting or explicitly biased policy tests are separately required.
5. **Quality:** fixed K2 shared-orthogonal vs matched backprop 8/16/32-update schedules, source-disjoint heldout CE and tool-call exactness. A predictive direction label is not the same as a demonstrated heldout CE gain.
6. **Authority:** classifier weight training, altered estimator sampling, GPU training, checkpoint promotion, deployed inference, hosted calls and NAS writes remain `DENY` unless separately approved with resource bounds and independent acceptance.

**Verdict:** **Real-model observation + independent PRE receipt and derived-label join: PASS. Independent heldout dataset size, classifier superiority and optimizer deployment: HOLD.**
