# Dust/K2 R2.4 — Cross-corpus quarantine enforced in classifier readiness

**2026-10-09 | Research-only draft | Stacked on [PR #28](https://github.com/scottjoyner/auto-finetune/pull/28)**

## Result

The earlier cross-corpus lexical scan in [PR #25](https://github.com/scottjoyner/auto-finetune/pull/25) could report overlapping source prompts and entire affected near-duplicate clusters, but the 16D readiness gate had no required consumer of that report. A previously contaminated source could therefore still appear in readiness capacity if the operator treated the lexical report separately.

**PR #29 closes that software gap** by adding a strict, optional SHA-pinned lexical-audit **veto** to the existing source-cluster readiness tool. It is a **read-only eligibility filter**, not a training-authority capability. It cannot assert that a lexical-negative corpus has passed semantic or cross-dataset independence, and it does not attest receiver signing-key custody.

## Changes

1. `k2_cross_corpus_audit.py` now binds a completed producer-local lexical scan to the **exact SHA-256 of the approved source preflight** and the pinned model-configuration digest. Missing auxiliary inputs no longer produce a misleading clean audit. Each report explicitly records its source: `PRODUCER_LOCAL_LEXICAL_AUDIT_ONLY`.
2. `k2_crosscorpus_veto.py` consumes a private SHA-pinned lexical report and the exact source preflight. It rejects missing or altered hashes, wrong source/model revision, inconsistent auxiliary truncation, duplicate or malformed cohort indices, unknown scanned files, reporting a mixed cluster as only partially quarantined, contradicting contaminated partition counts, and fabricated independent-source/receiver or training claims. It recomputes quarantine *propagation* across every member of the same near-duplicate family.
3. `k2_feature16_readiness.py` can now accept `--cross-corpus-audit` and `--cross-corpus-audit-sha256`. If a cluster was flagged by a valid audit, it is excluded from available eligible capacity. A previously exported 16D episode from that cluster is **rejected**, not silently counted as a clean training example. The aggregate report distinguishes `NOT_RUN`, `INCOMPLETE_TRUNCATED`, and `COMPLETE_FOR_SUPPLIED_CORPORA_ONLY`. All statuses preserve `real_label_classifier_training_authorized=false` and `independent_audit_of_cross_corpus_matching=false`.
4. Focused stdlib CPU tests reject an observed contaminated episode, rehashed partial-cluster forged quarantines, mismatched SHA pins, zero auxiliary corpora and truncated audit claims, while proving that a clean **lexical** result never authorizes semantic independence or classifier fitting.

## Acceptance and evidence

- [PR #29](https://github.com/scottjoyner/auto-finetune/pull/29), stacked on #28.
- Exact-head initial [cross-corpus veto CPU workflow](https://github.com/scottjoyner/auto-finetune/actions/runs/37960049038) **PASS**.
- On that same initial PR head, [round-two preregistration CPU](https://github.com/scottjoyner/auto-finetune/actions/runs/37960048988) **PASS** and [16D readiness CPU](https://github.com/scottjoyner/auto-finetune/actions/runs/37960048892) **PASS**.
- **No additional live Xwing source corpus was opened in this acceptance**: Fleet Commander denied host-inspection/process calls in this turn. No new cross-corpus leakage counts, model weights, classifier training or service credentials are claimed. This test establishes **correct behavior of the quarantine consumer**, not that approved corpora contain sufficient unique source groups.
- The previous actual 16D K2 export remains **one train source episode, eight direction rows**, file SHA256 `6935c43de2dcffad1405e1e10f009b3aaeb40404ff4faa0f483c618d470c804a`. Do not treat the preceding eight-episode *v1* source collection as a v2 16D dataset.

## Reproduce the read-only, private research gate (after approved local audit)

On Xwing, run the existing cross-corpus audit on approved and explicitly pinned auxiliary JSONL files, with a mode-600 output in a mode-700 SSD directory. Then calculate its SHA-256 and run:

```bash
python -m experiments.dust.k2_feature16_readiness \
  --read-only-readiness \
  --features /private/historyv2-smoke8.features16.v1.jsonl \
  --source-preflight /private/cohort128-preflight-v2.json \
  --source-preflight-sha256 18b624879eb5b265d2117bd76426b60307bba2e534dd924496a534a07b9d7b97 \
  --expected-model-sha256 6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365 \
  --cross-corpus-audit /private/NEW-pinned-cross-corpus-audit.json \
  --cross-corpus-audit-sha256 THE_ACTUAL_NEW_AUDIT_SHA256
```

The cross-corpus audit *must* be generated from the same exact source preflight hash; an older legacy report without this explicit binding cannot pass the new gate. No pre-existing file should be overwritten.

## Remaining independently required gates

1. **Receiver key custody HOLD:** existing Xwing SSH identity shares the `scott` x1 Unix principal and can read the old signing key. [PR #27](https://github.com/scottjoyner/auto-finetune/pull/27) includes a new restricted separate-UID service design, but it is **not deployed**. Operator approval and an actual negative key-read/privilege-escalation test from Xwing are required. Old receipts remain consistency-only evidence.
2. **Source adequacy HOLD:** need at least 64 train / 16 validation / 32 untouched test near-duplicate-disjoint source groups. Current v2 evidence has **one source group**; previous first-64 source window had 28/11/13 eligible clusters before a complete cross-corpus/semantic audit.
3. **Lexical scope HOLD:** scanning at SequenceMatcher >=0.85 is not an exhaustive paraphrase or semantic contamination test. The scanner caps auxiliary inputs and flags incomplete coverage; further approved offline semantic screens are required.
4. **Classifier training / unbiased gradient / deployment DENY:** the earlier synthetic learned model failed independent baseline confirmation. No real predictive classifier fit or learned direction selection should occur before independent dataset, heldout, trusted custody and matched-cost CE tests.

**Disposition: QUARANTINE-TO-READINESS LOGIC PASS; REAL EXTERNAL CORPUS ACCEPTANCE NOT RUN; SIGNER CUSTODY AND CLASSIFIER TRAINING HOLD.**
