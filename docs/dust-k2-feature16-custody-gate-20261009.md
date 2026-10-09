# K2 R2.2 — 16D causal-feature dataset gate and receiver-identity isolation

**2026-10-09 | Research-only draft | Stacked on [PR #26](https://github.com/scottjoyner/auto-finetune/pull/26)**

## Implemented

This slice makes the next classifier input explicit and auditable while **preserving the existing orthogonal estimator, original v1 derived labels, model weights and all live training gates**:

1. `k2_feature16_contract.py` requires a **complete independent CPU causal replay** of the existing K2 PRE/POST hash chain and derived-label evidence, before *any* 16D research-row export. Every candidate gets eight v1 geometric/directional scalars and eight v2 prior-completed-history scalars, in a frozen order. The first K=4 batch has all-zero history. Rows are written only on request into a new, mode-600 file in an existing mode-700 directory. No raw prompt/token/direction/activation vectors enter the output.
2. Real source splits are **provisional** based on the existing keyed-HMAC prompt group. The output explicitly says `receiver_key_isolated=false`, `source_clusters_independently_verified=false`, `classifier_training_authorized=false`. No classifier-fit CLI is provided.
3. `k2_restricted_receipt_service.py` is a **fail-closed deployment design**, requiring a completely separate x1 Unix service principal, private 0700 data root and mode-600 signing key owned by that service, root-owned nonwritable installed code and imported modules. It accepts only `SSH_ORIGINAL_COMMAND='receive <32 lowercase hex> <64 lowercase hex>'`. It rejects arbitrary paths, setup, verify, shells, forwarding requests and unprovisioned UID constants. It cannot run with the existing shared `scott` principal.
4. `IsolatedSSHReceiver` is an **optional, not-yet-deployed** client using explicit `dustreceipt` SSH user, a dedicated mode-600 client identity file, strict host-key checking, no forwarding or agent forwarding, and a forced receipt-only original command. The real-model runner accepts `--isolated-receiver-key-file` only with an explicit run ID. It **does not automatically claim separate key custody**, even when a response returns a valid-looking HMAC.
5. The inherited `k2_receipt_receiver.verify` previously returned unconditional `producer_has_receiver_key=false` and `independent_custody_for_precommit=true`. These claims were unsafe because Xwing could read the receiver key through its `scott` SSH identity. **They now return `producer_has_receiver_key=null` and `independent_custody_for_precommit=false`**, preserving cryptographic *ledger-consistency* as the only verified legacy claim.

### Feature order and limitations

Eight geometric v1 scalars: reserved zero for unavailable momentum, direction-to-anchor projection, direction-to-prior projection, quadratic activation spread, sigma-scaled spread, absolute spread-direction magnitude, sigma, and reserved zero for unavailable history.

Eight historical v2 scalars: fraction previously completed, prior beneficial rate, prior plus-gain mean/std, prior antithetic-slope mean/std, prior absolute-slope mean, prior latest plus gain. All historical signals derive only from **earlier completed** K2 antithetic forward calls. They are shared within the same four-direction batch, and alone cannot rank candidates. The v1 geometry still has **two zeros/unavailable predictors**: the feature schema is correctly auditable but its predictive sufficiency is not established.

A successful export requires a valid preexisting PRE chain, complete source-group HMAC and real-model revision match, matching original derived scalar labels, a fully observed K population, historical replay without leakage, and valid geometric slot semantics. Invalid or late features, unexpected tensor-like record fields, wrong source, malformed labels, unsafe permissions, existing destination or incomplete evidence fail closed.

## Receiver deployment: NOT EXECUTED

The Fleet Commander interface **explicitly blocks account/privilege management commands**. No new UID, authorized key, root-owned program, service secret or SSH policy was provisioned by this research slice. It would be misleading to claim isolated key custody at this stage. The operator must:

- Provision a **separate non-root receipt-only UID** on x1 with no producer shell access. Give it a private `/var/lib/dust-receipt` directory, owned by that UID and mode 0700, with a **new key** owned by the UID and mode 0600. **Do not migrate or reuse the legacy key.**
- Install an **immutable root-owned** copy of the receiver package at `/opt/dust-receipt/lib/experiments/dust/*`, with no producer/group write permissions in any ancestor directory. The pinned `INSTALLED_PATH` in the restricted module must match the installed copy. Provision a root-owned standalone bootstrap at `/usr/local/libexec/dust-receipt-entry` that starts a fixed Python interpreter in isolated mode (`python3 -I`) and imports this immutable package path; no producer-supplied `PYTHONPATH`, `PYTHONSTARTUP` or import path can be accepted.
- Associate exactly one producer-controlled **public** SSH key with that service account. The authorized-keys option should restrict *all* forwarding, PTY and shell capabilities and use a fixed command, conceptually: `restrict,command="/usr/bin/python3 -I /usr/local/libexec/dust-receipt-entry"`. The forced command must pass only the two permitted hex fields through the secure parser. Bind receiver service identity and producer UID **as immutable install-time constants** in its root-owned copy. The supplied research module defaults to `-1` and intentionally refuses to run before this separate deployment.
- Confirm from **Xwing using the exact existing producer SSH credentials** that the new key cannot be read, and no privilege escalation is possible to the service UID or root through those credentials. The existing ability to read the *old* key remains a separate legacy problem but must never allow the producer to read the **new** key. If the producer can exercise root/sudo through x1, a separate OS UID is insufficient; choose a different trust domain/hardware-backed signing service instead.
- Confirm a valid PRE digest can be signed through the dedicated `dustreceipt` forced command, while `whoami`, `verify`, file reads, arbitrary shell metacharacters, path arguments, agent/port forwarding, and repeated/out-of-order receipts are rejected. Independently rejoin new signatures with the PRE/RECEIPT/POST/derived files and verify a forged label fails. **No legacy signatures can become independent retroactively.**

The existing shared-principal x1 SSH authorization and global fleet services must stay unchanged until independently reviewed. Merely producing a HMAC receipt does not prove true numerical forward inference or semantic held-out source independence.

## Existing evidence and expected next experiment

Prior [PR #26](https://github.com/scottjoyner/auto-finetune/pull/26) produced a bounded real K2-Horizon-0.9B observation on Xwing with **8/8 directions, 2/8 beneficial plus scores, complete producer-local causal replay and zero model/adapter updates**. Immutable local-only source evidence hashes:

- Events `1899f4baaaac8cbca05ee31e4a4cfb84de020285217499c0421125331b83e899`.
- Derived labels `591d505696ce883b408f34c186c06bc15256648c0160e1d1cd944ae198660bfd`.
- Summary `a56ab076c46f48690194ced915fe0359ee30c338b0fcfb057ddadc14234e7ec6`.

This slice does **not** claim a new real-model run, a new real 16D export from Xwing, isolated receipt signing, or classifier training. Focused CPU tests are separate from previous real GPU evidence and cannot assert those unavailable gates.

The next confirmed sequence is: live service-UID/key isolation with negative read/privilege tests; privacy-reviewed and SHA-pinned cross-corpus/semantic contamination screening; >=64 train, 16 validation and 32 untouched source-disjoint held-out episodes; then a preregistered fit and paired source-cluster bootstrap comparison against momentum/curvature controls. Source groups, not direction rows, are the independent sampling unit. A classifier gain still cannot authorize biased gradient sampling or K2 optimizer deployment without separate matched-cost and held-out CE validation.

**Decision:** 16D data preparation and deny-only receipt security software: pending exact-head focused CI; trusted independent receiver custody: **HOLD**; real classifier training: **DENY**; production optimizer/sampling: **DENY**.
