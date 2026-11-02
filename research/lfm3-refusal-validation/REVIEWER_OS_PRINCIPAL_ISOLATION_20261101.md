# LFM3 XSTest reviewer isolation: verified Unix principal blocker (2026-11-01)

## Verified live read-only finding

The original 450-prompt/3-checkpoint review source packet and the thirty private reviewer batches remain on x1-370, all owner-only. Both `reviewer_1` and `reviewer_2` directories and the unblinding `private_model_key.json` share Unix UID 1000. The batch parent directory is 0700; each source or rating file is 0600. Thus two differently named `reviewer_id` fields DO NOT prove two independent Unix users or that either reviewer was prevented from reading the model key. No real human grades have been entered (0/1350 per assignment).

The read-only `src.reviewer_principal_audit` checks all original custody invariants, the actual filesystem owners/permissions of reviewer assignment folders and every original batch, and the key/source owners. Its status is aggregate-only and does not print prompt text, source IDs, owner usernames or grades. `--packet-dir` and `--batches-dir` use the actual private local paths; the guard `require_separately_isolated` reaudits the paths rather than trusting a caller-supplied boolean. Current independent reviewer access is ALWAYS marked **NOT VERIFIED**; no unsupported auto-provisioning or identity assertion is made.

## Why chmod or distinct reviewer IDs is not a fix

Changing `reviewer_1` and `reviewer_2` contents to different owners inside the **shared 0700 UID-1000 parent** would make the current on-host terminal unable to traverse the directories. Loosening the shared parent or model-key file permissions would break privacy. Likewise, generating two reviewer ID strings or receiving completed CSVs from one account does not prove real human independence.

## Required implementation boundary (not yet deployed)

1. The administrator, through an approved privileged operational change, provisions or selects **two distinct actual human accounts** with separately verified ownership/credentials. Do not have this agent invent identities, passwords, or act as either person.
2. Build **separate reviewer-specific owner-only workspaces**, each containing exactly that reviewer's 15 blinded batch CSVs and an independently verifiable manifest of frozen `review_id`, prompt and model response. **Neither workspace may contain** the model arm mapping, opposite reviewer's material, ground-truth safe/unsafe list, or source model weights. Do not place separate accounts under the original shared 0700 batch parent.
3. Provide a reviewer terminal capable of reading ONLY its own workspace, atomically recording stage-one human answers, resuming after interruption, and returning evidence with immutable source-content fingerprints. Do not reveal benchmark safe/unsafe eligibility until that reviewer completes all first-stage ratings.
4. Build an **owner-only, fail-closed import** of each independently completed workspace. Check its source hashes, unique reviewer identifier, complete status and file custody before attaching it to the canonical private packet. Record distinct Unix principal evidence and a human identity attestation; a self-reported username is insufficient. Failed or partial imports must never overwrite existing human ratings.
5. Only then permit the phase-two benign-helpfulness handoff and third-human correction/adjudication workflow. The current `src.reviewer_adjudication` now reports distinct human-ID STRINGS separately from independently verified OS principals and independently verified humans, and continues to prohibit model release.

## Reproduction (read only, aggregate only)

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PYTHONDONTWRITEBYTECODE=1 /home/scott/lfm3-research/train-venv/bin/python -m src.reviewer_principal_audit \
  --packet-dir /home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private \
  --batches-dir /home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
```

Expected current status: `reviewers_share_one_unix_uid=true`, `reviewer_folders_distinct_from_model_key_uid=false`, `independent_reviewer_access_verified=false`, `physical_independence_of_two_humans_verified=false`, `release_authorized=false`. Synthetic contract tests cover source custody, same UID, unsafe permissions, key alias, and fail-closed identity gate.

**Do not claim human-independent review or release readiness from the current single-UID setup.** PR #103 stays draft, original frozen checkpoints stay unchanged, storage issue #106 stays open.
