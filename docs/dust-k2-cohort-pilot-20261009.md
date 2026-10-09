# Dust/K2 source-cohort preflight + read-only classifier-label pilot

**2026-10-09 | High priority | RESEARCH DRAFT | Parent: [PR #23](https://github.com/scottjoyner/auto-finetune/pull/23), classifier [PR #22](https://github.com/scottjoyner/auto-finetune/pull/22) and Dust/K2 [PR #13](https://github.com/scottjoyner/auto-finetune/pull/13)**

## Decision in one sentence

The real K2 observer now collected **64 actual antithetic direction labels over 8 lexically disjoint source clusters** (4 train / 2 validation / 2 test), with matched producer and x1 receiver HMAC event chains, while **classifier fitting remains DENIED** due to dataset underpowering, unvalidated historical features, incomplete semantic contamination checks and a demonstrated shared-SSH signing-key custody gap.

## Why this slice was necessary

The former read-only real K2 experiment yielded 64 directions from **one** source, which is just **one independent episode**, not 64 training cases. The first audit of the existing 254 MiB `train.combined.jsonl` at 96-token limit found **317 unique prompt/response pairs but only 47 tokenizable pairs, 41 unique exact prompts**, with repeated near-duplicate responses. Never mix response variants across source-heldout partitions.

At 128 tokens and under the existing pinned local tokenizer, **75** tokenizable pairs were found, grouped into **62 near-duplicate lexical clusters** using normalized raw prompt text and conservative SequenceMatcher ratio **≥0.85**. The collector's first 64-index window contains **28 eligible train clusters, 11 validation, 13 test**, with **one quarantined cluster whose near-duplicate members spanned different split assignments**. Nine additional clusters lie beyond the current limited source-index window. There are 62 eligible *sample indices* but NOT 62 independent collected episodes.

The new `k2_cohort_preflight.py` runs entirely on Xwing where prompt text already resides. No prompt strings, token IDs or responses are written to the JSON manifest. It emits masked-prompt HMACs and a separate privacy-opaque SHA256 cluster ID, permanently marks cross-split near-duplicate clusters `QUARANTINE`, pins dataset SHA, model config SHA, token cap and source index. This is a **lexical, within-one-corpus** duplicate screen: semantic paraphrases and cross-dataset contamination are still NOT VERIFIED.

The real `k2_real_direction_probe.py` now **requires the exact pinned preflight SHA256 and matching sample/source-group HMAC before loading pretrained weights**. No opt-out in the research CLI. `k2_cohort_collect.py` selects exactly one source episode per cluster, refuses quota overrun, requires mode-700 local evidence directory, observes all K directions and requests fresh x1 PRE acknowledgments per source. The initial bounded cohort uses **8 directions per episode**, not 64, to prioritize source diversity while preserving complete counterfactual labels.

## Observed Xwing actual pretrained K2 run

**Platform:** Existing pinned K2-Horizon-0.9B on Xwing Radeon 8050S, ROCm Torch 2.12.0, local cached `train.combined.jsonl`. No model/tokenizer downloads, hosted calls, NAS writes, checkpoints, LoRA updates, backprop or production scheduling. All files on existing Xwing SSD mode 600 (directory mode 700). x1 received a private copy of scalar evidence, never prompts or weights.

**Pinned model weights SHA256:** `6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365`

**Source dataset SHA256:** `2b7b01b0388474af9255c62de0be28aaedb26af2494c25f3bc73493d050e8426`

**Model config SHA256:** `0ba8f6a0fe8daa5003f88c335735cabc7dba20600ace939efab949ae5e59b936`

**Preflight v2 manifest SHA256:** `18b624879eb5b265d2117bd76426b60307bba2e534dd924496a534a07b9d7b97`

**Original producer cohort summary SHA256:** `71a851eae4c66454bac2da0b0367b2f421b96de0cc696231f6ed37e5928b75e0`

| Partition | Source clusters observed | Completed direction rows |
| --- | ---: | ---: |
| Train | 4 | 32 |
| Validation | 2 | 16 |
| Test | 2 | 16 |
| **Total** | **8** | **64** |

All 64 probes completed with 16 PRE-batch HMAC receipts, unique source clusters and unchanged base + LoRA parameters; **20 of 64** had lower plus-direction loss than the clean state. The 20/64 statistic measures labels across eight specific sources—not held-out CE improvement or predictive classifier quality. Exactly **8 independent source episodes**, not 64.

Evidence files on Xwing:

- `/media/scott/data/finetune-staging/research-witness-20261009/cohort128-preflight-v2.json`
- `/media/scott/data/finetune-staging/research-witness-20261009/cohort-v2-pilot-8/cohort-collection-summary-v1.json`
- Unique `events.jsonl`, `derived.jsonl`, `summary.json` and restricted stderr audit log per observed source.

## x1 receiver audit and important custody correction

x1 physically stores receiver-owned HMAC signing material and hashed batch PRE receipts. Its local `k2_cohort_reconcile.py` re-verifies every copied event hash, derived label file digest, summary SHA, batch receipt HMAC chain, all PRE/RECEIPT/POST order and precise derived scalar fields. It also verifies eight distinct near-duplicate clusters, group assignments and model revision. The batch audit **PASS** found **16/16 HMAC receipts** and **64/64 labels** matched the original PRE/POST evidence.

**Critical security issue discovered:** Xwing can SSH into x1 as the **same `scott` Unix user**, and this remote account could **read** `/home/scott/git/dust-k2-receiver-custody-20261009/receiver-owned.key` despite its mode 600. The key lives physically on x1, but the producer *can access the signing credentials* via its general-purpose SSH privileges. Accordingly the current receipts are **verified cross-node ledger consistency, NOT cryptographically independent producer-resistant custody**. This must be corrected before independent custody acceptance. We did NOT modify fleet SSH permissions or rotate shared user credentials in this research PR.

**Revised x1 reconcile result:** `PASS_LEDGER_INTEGRITY__KEY_ISOLATION_UNPROVEN`; the `HOLD_SHARED_UNIX_PRINCIPAL` trust gate remains. `producer_cannot_read_receiver_key=false`. Complete revised report SHA256:

`5bb9816e58b5c1b14707d9bf4cbe13daedff0eda0a96a0ea1ddeb850df2b169b`

Original first-pass reconciliation is preserved rather than overwritten. The revised report is at `/home/scott/git/dust-k2-receiver-custody-20261009/cohort-v2-pilot-8-intake/reconciliation-v2-key-isolation-hold.json`.

A credible next key-isolation implementation must use a **separate, restricted receiver service identity** (no producer shell or general read access to the signing key), locked-down forced-command/service endpoint with rate limits, and demonstrable denial of reads under the *same producer SSH credentials*. Preserve existing fleet administrative access unless explicitly planned. HMAC receipt signing alone also does not prove K2 loss numerical correctness.

## Tests and reproducibility

- 14 focused **standard-library CPU tests on x1** pass at implementation head prior to documentation/CI changes: near-duplicate quarantine, source drift, cluster uniqueness, group-aware train/val/test quotas, no raw prompt serialization, malicious receipt-copy mutation, source replay, private file permission rejection and explicit receiver key-isolation HOLD.
- The actual pretrained model observed eight real source groups on Xwing with fresh separate run IDs; no prior source-grouping records were silently reused or promoted.
- A focused GitHub Actions job tests the stdlib dataset admission and receiver reconciliation code without installing Torch or downloading any weights. The broader full-project test workflow has independent pre-existing dependency/coverage issues and is not claimed green.

### Exact local preflight

```bash
python -m experiments.dust.k2_cohort_preflight --read-only-preflight \
  --model-dir /media/scott/data/finetune-staging/models/K2-Horizon-0.9B \
  --train-jsonl /media/scott/data/finetune-staging/data/datasets/train.combined.jsonl \
  --episode-key-file /media/scott/data/finetune-staging/research-witness-20261009/episode-hmac.key \
  --output /path/to/NEW-private-cohort-manifest.json
```

The collection CLI requires that manifest's exact SHA and offers bounded 4/2/2 source quotas. `k2_cohort_reconcile.py --verify-only` runs on x1 against an imported private evidence directory and the receiver's own receipt ledger, **not** against producer self-claims alone.

## Next science/engineering acceptance gates

1. **Receiver service separation:** production of false PRE receipts and receiver-key reads via Xwing's principal must be demonstrably denied. Until proven, `independent_receipt_key_custody=false`.
2. **Larger and cleaner source corpus:** current first-64 pilot has only 13 eligible test clusters. Obtain enough genuinely independent source groups from approved local corpora, unify HMAC scope and deduplicate *across* corpora plus semantic paraphrases. Keep all sources from the same near-duplicate family in one split. Requirement of ≥32 true independent held-out source episodes remains unmet.
3. **Real classifier input schema:** first real direction witnesses carry geometric proxy features with placeholder historical momentum features. Those are **not equivalent** to the toy synthetic classifier input distribution, and should not be mislabeled or reused as a training-ready feature set. Add historical measurements only if truly preceding the candidate's PRE timestamp.
4. **Classifier fit only after data and custody gates:** train 3-seed logistic/MLP baselines on source-disjoint train groups with validation-only tuning, then once on an untouched ≥32-group test cohort. Compare top-K gain/precision/Brier/calibration against momentum, curvature-aware momentum and random baselines; report paired bootstrap over **source clusters**. The synthetic classifier failed its independent confirmation; do not promote it based on this small pilot.
5. **Preserve unbiased estimator:** any eventual selection changes gradient sampling. Require matched compute/counterfactual K controls and explicit importance-weighted/unbiased-gradient checks or clearly marked biased selection experiment, plus held-out K2 CE and tool-call quality.

**Conclusion:** Real multi-episode acquisition and ledger integrity **PASS**. Independent signing-key custody, scientifically sufficient heldout corpus, validated real-history classifier features, real model classifier fit and production authority all remain **HOLD**.
