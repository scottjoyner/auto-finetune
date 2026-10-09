# Immutable experiment run card (copy once per planned run)

> Template only. Freeze the **Predicted** section and commit/hash it before any instrumented run. Do not rewrite old results; make a new run record for retries or amendments.

## Header
- Run ID: `NOT_ASSIGNED`
- Preregistration version/commit: `NOT_LOCKED`
- Stage/arm: `NOT_RUN`
- Parent experiment / previous attempts: `NONE`
- Date prepared UTC / date executed UTC: `NOT_SET` / `NOT_RUN`
- Executing agent, supervising human, independent reviewer: `NOT_ASSIGNED`
- Approved workdir / physical host / resource lease / expiration: `UNAPPROVED`
- GPU and provider authorization: `DENY` / `DENY`

## Frozen assets and permits
- Upstream LittleBit commit, license snapshot and reviewed use: `UNVERIFIED`
- Model, config, revision, tokenizer SHA-256: `UNVERIFIED`
- Dataset training/calibration/validation/heldout/final test manifests and separation proof: `UNVERIFIED`
- OS/kernel/runtime/driver/pytorch/transformers/CUDA-or-ROCm/commit hashes: `UNVERIFIED`
- Requested max CPU RAM, GPU VRAM, local disk, GPU seconds and network egress: `0 / NONE`
- Baseline+control run IDs: `NOT_ASSIGNED`

## Prediction (commit before job)
- Explicit H0 / H1, expected sign and size of effect: `NOT_LOCKED`
- Fixed bit budget (block vs whole-model), rank and residual allocation: `UNVERIFIED`
- Fixed primary metric and heldout split: `NOT_LOCKED`
- Decision threshold and stopping rule: `NOT_LOCKED`
- Run command hash / config hash and matching controls: `UNVERIFIED`

## Observations (must remain NOT_RUN until measured)
| Metric | Predicted | Observed | Unit / lineage |
| --- | --- | --- | --- |
| Reconstruction error | UNSET | NOT_RUN | relative Frobenius |
| Heldout token-weighted CE | UNSET | NOT_RUN | nats/token |
| Agentic task pass rate | UNSET | NOT_RUN | success/attempts |
| Actual compressed model storage | UNSET | NOT_RUN | bytes |
| Measured effective block BPW | UNSET | NOT_RUN | bits/original block param |
| Total packaged model bytes | UNSET | NOT_RUN | bytes |
| Training max allocated/peak GPU | UNSET | NOT_RUN | bytes |
| Full+tail forward counts | UNSET | NOT_RUN | integer |
| Full model backward calls | UNSET | NOT_RUN | integer |
| Wall time/energy | UNSET | NOT_RUN | seconds/joules when credible |
| Decode/prefill TPS | UNSET | NOT_RUN | tokens/sec same-node |

## Failures, lineage and sign-off
- Model data or code changed midrun? `NOT_RUN`
- No unexpected provider requests/deployment/NAS writes? `NOT_RUN`
- Logs/data reviewed for raw prompts/secret leakage? `NOT_RUN`
- Terminal status and exit code: `NOT_RUN`
- SHA-256 hashes of aggregate public-safe artifacts: `NOT_RUN`
- Original prediction supported / refuted / inconclusive: `NOT_ASSESSED`
- Reviewer decision GO / HOLD / FAIL: `HOLD`
- Narrowest next justified experiment and fresh approval: `PENDING`
