# K2 R2.3 — real 16D source-cluster readiness and experiment capacity

**2026-10-09 | RESEARCH / DRAFT | Stacked on [PR #27](https://github.com/scottjoyner/auto-finetune/pull/27)**

## Completed real-data milestone

The new 16D exporter from #27 was run on Xwing against **actual pretrained K2-Horizon-0.9B** producer-local v2 history evidence created in [PR #26](https://github.com/scottjoyner/auto-finetune/pull/26). The replay-first export succeeded with **8 real directions and 16 immutable features per direction**, without model inference, training, checkpoint or any NAS operation in this slice.

**Private actual exported file:** `/media/scott/data/finetune-staging/research-witness-20261009/historyv2-smoke8.features16.v1.jsonl` (mode `0600`, 6,781 bytes).

**Export SHA256:** `6935c43de2dcffad1405e1e10f009b3aaeb40404ff4faa0f483c618d470c804a`. Not uploaded to GitHub; keyed source-group identifiers and individual loss statistics remain private to Xwing.

Read-only `k2_feature16_readiness.py` subsequently validated that real export against the exact pinned 128-token source preflight and pretrained model hash:
- preflight SHA256 `18b624879eb5b265d2117bd76426b60307bba2e534dd924496a534a07b9d7b97`;
- model SHA256 `6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365`;
- **8 complete directions from ONE train-partition source cluster**; **2 beneficial plus-direction labels**;
- 16 feature columns, **13 varied within this single episode** (does NOT establish predictive information or independence);
- training source-cluster counts `train=1, validation=0, test=0`, far short of frozen `64/16/32`;
- read-only gate `HOLD_NO_INDEPENDENT_CUSTODY_OR_SUFFICIENT_SOURCE_GROUPS`;
- `real_label_classifier_training_authorized=false`, `optimizer_or_production_authorized=false`.

The historical real-model loss and feature data are pre-existing producer-local research evidence. This slice ran a **file validation and export only**, not another model inference/training pass.

## New acceptance code

`experiments/dust/k2_feature16_readiness.py` consumes private feature16 records and a SHA-pinned, private preflight manifest. It refuses an unsupported population, duplicate/extra/unsafe JSON fields, NaN or Infinity, noncontiguous candidate indices, mixed prompt HMACs, model revision drift, fabricated provenance, incompatible geometry constants, inconsistent positive-gain labels, first-batch future history, intra-batch history variations, unsafe file modes or symlink files, altered cohort hash, quarantined source groups, duplicated prompt groups, or reused near-duplicate clusters.

It counts independent **source clusters**, not direction rows. Its aggregate report contains only group counts, positive labels, provisional split distribution, feature-column variation and the remaining group deficits. It does not print prompt HMACs, real 16D vectors, source text, or individual losses.

A second enhancement estimates **eligible capacity** in the exact preflight, distinct from **actually observed** source groups, and rejects cross-partition reuse of an eligible near-duplicate cluster. This identifies whether more approved and deduplicated source corpora are necessary *before* spending additional K2 ROCm GPU time.

Historical single-corpus preflight on Xwing found, in the bounded first-64-index window, **28 train / 11 validation / 13 test** distinct eligible clusters (with a mixed-split near-duplicate family quarantined). Against the frozen **64 train / 16 validation / 32 heldout** requirement, that window alone lacks at least **36 training, 5 validation and 19 heldout clusters**, even if every otherwise eligible group were probed. A new live check of the capacity enhancement is pending exact-head validation; do not treat historical cluster numbers as a completed cross-corpus or semantic independence audit.

## Commands, read-only local acceptance

```bash
cd /media/scott/data/git/wt-dust-k2-feature16-readiness-20261009
python -m experiments.dust.k2_feature16_readiness \
  --read-only-readiness \
  --features /media/scott/data/finetune-staging/research-witness-20261009/historyv2-smoke8.features16.v1.jsonl \
  --source-preflight /media/scott/data/finetune-staging/research-witness-20261009/cohort128-preflight-v2.json \
  --source-preflight-sha256 18b624879eb5b265d2117bd76426b60307bba2e534dd924496a534a07b9d7b97 \
  --expected-model-sha256 6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365
```

This command prints aggregate-only readiness; never uses GPU, model inference, network or hosted inference, never writes a corpus, and can be evaluated with private local data. The code offers **no** classifier training or production operation.

## Explicit HOLD boundaries

1. **Independent receipt signing remains HOLD.** Legacy Xwing SSH as `scott` could read the previous x1 receiver HMAC key. [PR #27](https://github.com/scottjoyner/auto-finetune/pull/27) drafted a separate-UID receipt-only service, but that privileged service and a negative producer key-read/access escalation test are NOT deployed/verified.
2. **Source adequacy remains HOLD.** The evidence has **one real v2 feature episode**. Earlier 8-group cohort collection from [PR #24](https://github.com/scottjoyner/auto-finetune/pull/24) produced **v1-only feature evidence**; do NOT silently treat it as eight v2 16D episodes. Count source clusters as independent units. Require >=64 / 16 / 32 genuinely distinct and near-duplicate-screened source clusters for a classifier training/evaluation experiment.
3. **Full cross-corpus and semantic duplication screening remains NOT VERIFIED.** Earlier read-only auxiliary scan attempts were blocked. An in-corpus lexical screen alone cannot establish prompt disjointness from external datasets or paraphrased tasks.
4. **Classifier superiority remains unproven.** The original toy three-head mini-batch SGD ensemble lost to curvature-aware momentum on its frozen independent synthetic confirmation. The real v2 features cannot be promoted as a predictor until a new, source-disjoint, preregistered comparison beats the fixed baselines.
5. **Unbiased-gradient and held-out K2 CE remain NOT RUN.** Any learned direction selection changes sampling. The shared orthogonal estimator is untouched; no optimizer writes or production authority are granted.

## Test evidence / disposition

At initial implementation head on x1-370, **5/5 new readiness CPU tests PASSED**; actual Xwing private file was accepted with the correct science-deny gate. Follow-up preflight-capacity and cross-split-cluster denial tests were added and require exact-head CI verification. The focused CI job is `.github/workflows/dust-k2-feature16-readiness-cpu.yml`.

**Decision: REAL FEATURE16 EXPORT + READINESS SMOKE PASS; SOURCE COUNT, INDEPENDENT KEY CUSTODY, CLASSIFIER TRAINING AND PRODUCTION DEPLOYMENT HOLD.**
