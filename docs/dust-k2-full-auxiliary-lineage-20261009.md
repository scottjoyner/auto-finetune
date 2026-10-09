# Dust/K2 R2.5 — complete auxiliary lexical audits and source-lineage triage

**2026-10-09 | RESEARCH DRAFT | Stacked on [PR #29](https://github.com/scottjoyner/auto-finetune/pull/29)**

## Materially new live evidence

The earlier 512-pair audit cap was insufficient for Xwing's `general-norobots.jsonl` auxiliary corpus: **11,037 unique prompt/response pairs** existed, so screening only 512 left **10,525 unexamined**. This slice adds an opt-in bounded `--auxiliary-pair-cap 16384`; the default remains 512. `SequenceMatcher.quick_ratio()` is used exclusively as a sound upper-bound pruning step ahead of the unchanged lexical similarity threshold **0.85**. A covered file must report `truncated=false` and `pairs_screened==unique_pairs_reported`. Both the scanner and quarantine consumer reject declared caps above 16,384 or below 512.

All the below comparisons ran **on Xwing's existing local SSD, CPU/tokenizer only, no GPU model weights, no K2 forward/backward, no optimizer, NAS writes or hosted calls**. Evidence files remain mode 0600 in a mode-0700 research directory; no raw prompts, source-group HMACs, tokens or per-direction losses were sent to GitHub.

### Full lexical overlap against the existing pinned 75-example K2 cohort

| Existing auxiliary file | Unique paired records fully screened | K2 cohort candidate prompts with lexical overlap >=0.85 | Exact candidate prompt matches | Truncated? |
| --- | ---: | ---: | ---: | --- |
| `general-norobots.jsonl` | **11,037 / 11,037** | **0 / 75** | 0 | No |
| `train.mixed.jsonl` | **6,435 / 6,435** | **75 / 75** | **75 / 75** | No |
| `train.hermes-reasoning.jsonl` | **283 / 283** | **55 / 75** | **54 / 75** | No |

These compare **normalized user prompts**, not identical full paired responses, and only the 75 tokenizable training candidates rather than every source message. The 75/75 exact prompt containment of `train.mixed` strongly indicates source reuse/derivation; the classifier pipeline **must not count its record total as 6,435 novel independent episodes**. `train.hermes-reasoning` also overlaps extensively. Conversely, zero lexical matches for `general-norobots` is a potential new source worth examining, **not proof of source independence**, language-model quality, semantic nonduplication, or licensing rights.

**Pinned producer-local source preflight SHA256:** `18b624879eb5b265d2117bd76426b60307bba2e534dd924496a534a07b9d7b97`.

**Actual full auxiliary evidence files and SHA256s:**

- Xwing `/media/scott/data/finetune-staging/research-witness-20261009/cross-corpus-full-general-20261009.json` — `98a9cd82a6f51f4b5e05e15471b6ea801006936d4cd63d4bac757a665f5f771b`.
- Xwing `/media/scott/data/finetune-staging/research-witness-20261009/cross-corpus-full-mixed-hermes-20261009.json` — `9d8c9fe5870958d0ee8a99f03c38764bbc887c954398eff7c605a9b27ee4b5a8`.

Both reports are fully covered for their **listed** auxiliary files. They do NOT certify every corpus on the machine has been audited. They also do not establish semantic/near-paraphrase independence.

## Real fail-closed integration witness

Invoked the existing `k2_feature16_readiness` against the **actual previously observed 8-direction, 16-feature K2 evidence** (`historyv2-smoke8.features16.v1.jsonl`, SHA256 `6935c43de2dcffad1405e1e10f009b3aaeb40404ff4faa0f483c618d470c804a`).

- With full `general-norobots` evidence (no lexical overlap), the gate returned `COMPLETE_FOR_SUPPLIED_CORPORA_ONLY`, zero quarantined source clusters, **28/11/13** eligible source-cluster capacity under the original pinned pilot preflight and still `classifier_training_authorized=false`.
- With the fully overlapping `train.mixed` + partially overlapping `train.hermes-reasoning` evidence, the gate **rejected** that same K2 observation with `ValueError: source absent from eligible pinned cohort or quarantined by cross-corpus audit`. All 75 candidates are covered by the matched source lineage report, extending to 62 existing related near-duplicate clusters.

The different outcomes are scientifically correct for different audit scopes. **Do not treat a related training-data derivative as an independent external held-out corpus**; rather, preserve its lineage and exclude reuse as a new source. Do not overwrite or reclassify any original training/evidence files.

## New lineage triage code and tests

`k2_source_lineage_gate.py` combines up to six independently SHA-pinned private lexical audit reports, rechecks whole-cluster quarantine with the same source preflight, and classifies each underlying auxiliary file:

- `CONTAINS_ALL_CANDIDATE_PROMPTS_EXACTLY` — `train.mixed`, not a novel source pool relative to the candidate cohort;
- `SHARES_CANDIDATE_PROMPTS` — `train.hermes-reasoning`;
- `NO_DETECTED_LEXICAL_OVERLAP__NOT_INDEPENDENCE_PROOF` — `general-norobots`;
- also handles incomplete scans and corpora without compatible records.

The actual read-only lineage run on Xwing reported **3/3 complete lexical scans, 62 previously known overlapping candidate clusters when combining matched corpora, no new independent held-out authorization**. It exposes aggregate counts and corpus SHA256s only, never normalized user strings, response text or private source HMACs. Tests cover derivative classification, multiple audit SHA binding, duplicated source prevention, privacy, 512+ record matches, truncated/overstated cap refusal and preservation of exact ≥.85 similarity decisions.

## Tokenizer-only candidate-capacity experiment

The existing K2-Horizon tokenizer, with `max_tokens=128` and the current source pair reader, found:

- `general-norobots`: **11,037 unique prompt/response pairs**, of which **4,266 paired examples** passed the K2 128-token `tokenize_pair` contract.
- Those **4,266 candidate rows** correspond to **4,142 exact distinct normalized user prompts** (124 response-variant repetitions beyond one per exact prompt). One prompt had up to 42 valid response variants, demonstrating that paired-row counts overstate independent prompt count.
- No tokenizer exceptions were observed in the first full pass.
- These figures establish **potential tokenizable capacity**, not license/reuse approval, near-duplicate clusters, semantic independence, or verified train/validation/test splits.

This is a meaningful potential expansion beyond the previous bounded 75-example pilot, but any future source merger must deduplicate across the **entire** original training corpus, not only 75 128-token candidates, then cluster near/paraphrased prompts with a recorded false-positive/false-negative policy and versioned source-group HMAC ownership. Do not leak prompts or tokens in review artifacts.

## Remaining HOLDs (not waived by this result)

1. **Independent receiver key custody HOLD:** Xwing can access x1 legacy `scott` signing credentials; the draft separate non-root `dustreceipt` forced-command service in PR #27 has not been provisioned/independently verified. Existing HMAC receipts are ledger consistency only.
2. **Rights and independent-heldout data HOLD:** `general-norobots` rights, provenance, full-source/semantic duplicate screening and frozen source-disjoint split authority are unverified.
3. **Actual new K2 16D source episodes NOT COLLECTED:** the real K2 historical 16D exporter has **one** source group; the older 8-group observation is v1-only. The frozen acceptance remains at least **64 train / 16 validation / 32 untouched held-out** source clusters. No new model evaluation was performed here.
4. **Classifier, sampling, optimizer, production DENY:** synthetic classifier failed earlier independent comparison with curvature-aware momentum. Even an eventual successful feature-only classifier does not authorize changing stochastic gradient selection without matched-cost bias and heldout language-model CE evidence.

**Disposition: complete bounded lexical audits + source-lineage triage + actual veto enforcement PASS; classifier/data-custody promotion HOLD.**
