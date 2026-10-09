# Dust/K2 R2.6 — exhaustive local auxiliary prompt-family audit

**2026-10-09 | Research-only, draft | Stacked on [PR #30](https://github.com/scottjoyner/auto-finetune/pull/30)**

## What changed

The previously identified `general-norobots.jsonl` source held **11,037 unique prompt/response pairs**, of which **4,266** could be represented by the existing K2-Horizon-0.9B tokenizer's strict 128-token supervised pairing contract. Those pairs correspond to **4,142 exact-distinct normalized user prompts**. Pair counts cannot be treated as independent prompt episodes: **124 excess response variants** exist, and one exact prompt has **42** eligible response variants.

This slice implements a bounded, *exhaustive* connected-component calculation over those 4,142 eligible exact-distinct prompts at the previously frozen `SequenceMatcher(autojunk=False)` lexical similarity threshold **0.85**. It is not an embedding/semantic paraphrase screen.

- Unlike an approximate ANN/MinHash index, `k2_auxiliary_prompt_clusters.py` enumerates every unordered prompt pair, safely eliminating pairs only when (a) string lengths prove they cannot attain 0.85 matching ratio or (b) `SequenceMatcher.quick_ratio()` (an upper bound) proves they cannot pass. An actual `.ratio() >= 0.85` links two prompts, and transitive link closure places their entire related family in the same component.
- No raw source prompts, targets, token IDs, geometric features, model parameters or per-direction losses are persisted. The only row identifiers in the mode-600 private manifest are **keyed HMACs** of exact normalized prompt identity and lexical family. A family-level provisional split is assigned **after** grouping, never per response variant. No split is approved for training or held-out use.
- Hard bounds: up to **5,000** distinct exact normalized prompts; a **240-second** CPU cutoff; failure causes **no partial manifest**. `k2_auxiliary_cluster_reconcile.py` independently checks the pinned private file SHA, mode, source/model revision hashes, family uniqueness, split agreement, response-variant totals, pair coverage and all nonpromotion flags.
- `k2_full_source_overlap.py` extends the prior 75-tokenizable-example comparison to the **entire original 317-pair source corpus**, including pairs that cannot fit the strict 128-token K2 budget.

## Actual Xwing CPU evidence (no GPU inference)

| Metric | Actual result |
| --- | ---: |
| Auxiliary unique prompt/response pairs | 11,037 |
| K2 strict 128-token pairs | **4,266** |
| Exact-distinct normalized eligible prompts | **4,142** |
| Extra paired responses beyond one per exact prompt | 124 |
| Total possible unordered distinct prompt comparisons accounted | **8,576,011** |
| Safe length-pruned pair count | 6,250,833 |
| Length-eligible candidate pair count | 2,325,178 |
| Quick-ratio-rejected pairs | 2,319,676 |
| Matching lexical links at 0.85 or above | **183** |
| Exact lexical related-prompt connected components | **4,019** |
| Maximum distinct prompts in one component | 11 |
| Omitted pairs due to approximate nearest-neighbor indexing | **0** |

**Candidate-only family split counts:** training **2,415**, validation **768**, test **836**. These are *source-family capacity measurements*, not approved data splits, not 4,019 independently certified episodes, and absolutely **not observed 16-feature K2 training records**. They exceed the preregistered minimum cohort-size targets of 64/16/32 on paper, but semantic paraphrase screening, data rights, source quality, split acceptance and independent signer credentials must still be established.

**Private manifest:** Xwing `/media/scott/data/finetune-staging/research-witness-20261009/general-norobots-lexical-clusters-20261009.json`; mode `0600`, **1,917,223 bytes**; SHA256:

`69642839c28a2e74c0ea5e523424136ef3ca9451cd102244753aeeea4540d3cc`.

The independent Xwing verifier returned `PASS_PRIVATE_LEXICAL_ACCOUNTING_ONLY` and confirmed all **4,142** private HMAC prompt identities, **4,019** family identities, **4,266** response variants and identical provisional split counts. It also verified `classifier_training_authorized=false`, `dataset_reuse_rights_approved=false`, `receiver_key_custody_independent=false`, and no private prompts or HMACs in the aggregate verification output.

### Full original-source contamination comparison

The older 75-candidate source preflight was narrower than the entire `train.combined.jsonl` corpus. The new read-only scanner compared **all 317 unique original source prompt/response pairs** (**210 exact-distinct normalized source prompts**) with all **11,037** unique `general-norobots` paired entries (**10,912 exact-distinct prompts**).

Result: **zero exact and zero ≥0.85 lexical matches** at the full-source prompt level. This is stronger than the prior 75-example audit but does not prove semantic paraphrase independence or licensing/data rights.

**Private full-source report:** `/media/scott/data/finetune-staging/research-witness-20261009/full-original-vs-general-norobots-20261009.json`, mode `0600`, SHA256:

`26ff5dc34961eab75af8364c3839ff97251141b890a9b4ce5884fe3d90cbe68b`.

**Original source SHA256:** `2b7b01b0388474af9255c62de0be28aaedb26af2494c25f3bc73493d050e8426`.
**Auxiliary source SHA256:** `27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5`.
**Tokenizer config SHA256:** `0ba8f6a0fe8daa5003f88c335735cabc7dba20600ace939efab949ae5e59b936`.

## Reproduction (private, read-only; output exclusive-create)

```bash
cd /media/scott/data/git/wt-dust-k2-auxiliary-prompt-clusters-20261009
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m experiments.dust.k2_auxiliary_prompt_clusters \
  --read-only-private-cluster-audit \
  --source-jsonl /media/scott/data/finetune-staging/data/datasets/general-norobots.jsonl \
  --model-dir /media/scott/data/finetune-staging/models/K2-Horizon-0.9B \
  --episode-key-file /media/scott/data/finetune-staging/research-witness-20261009/episode-hmac.key \
  --cutoff-seconds 240 \
  --output /path/to/NEW-mode600-private-candidate-manifest.json
```

The published manifest already exists: choose a NEW path for any repeat rather than overwriting evidence. To verify the existing immutable file without access to the prompt corpus:

```bash
python -m experiments.dust.k2_auxiliary_cluster_reconcile --verify-only \
  --manifest /media/scott/data/finetune-staging/research-witness-20261009/general-norobots-lexical-clusters-20261009.json \
  --expected-manifest-sha256 69642839c28a2e74c0ea5e523424136ef3ca9451cd102244753aeeea4540d3cc \
  --expected-source-sha256 27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5 \
  --expected-model-config-sha256 0ba8f6a0fe8daa5003f88c335735cabc7dba20600ace939efab949ae5e59b936
```

All secret HMAC identifiers and source prompts remain on Xwing. Neither GitHub Actions nor the GitHub report includes the private manifest.

## Frozen remaining gates

1. **Source rights/provenance HOLD:** `general-norobots` local availability is not permission to reuse it for classifier training or public redistribution. Confirm provenance, licensing and retention first. Keyed HMACs use the producer's existing research secret, not independently isolated receiver credentials.
2. **Semantic dedup HOLD:** exact and SequenceMatcher ≥0.85 grouping is exhaustive for **that lexical rule**, but semantic paraphrases, translated prompts and task-equivalent requests may evade it. A second human-reviewed semantic screen and false-negative controls are required before held-out allocation.
3. **Train/validation/test split authority HOLD:** the 2,415/768/836 numbers are provisional **family-level buckets** only. Do not silently treat them as a trained-model/heldout-ready dataset. No K2 real direction probes from these families have been collected in this slice.
4. **x1 isolated receipt key HOLD:** Xwing can still read the legacy x1 HMAC key via shared `scott` SSH identity; the restricted separate-UID receipt-only service in PR #27 is not provisioned. No new signing-key acceptance occurred.
5. **Real classifier and optimizer DENY:** the existing full K2 observation contains just ONE 16D source episode, from the old source corpus. No classifier fitting, momentum-benchmark victory, biased sampling validation, checkpoint/LoRA update, NAS operation, hosted provider or production routing is authorized.

**Disposition: deterministic private source-family capacity and complete full-original-source lexical screen PASS; independent source/rights/custody acceptance and classifier fitting HOLD.**
