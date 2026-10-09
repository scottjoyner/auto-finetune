# Dust/K2 R2.7 — No Robots provenance and token-order overlap triage

**2026-10-09 | RESEARCH DRAFT | Stacked on [PR #31](https://github.com/scottjoyner/auto-finetune/pull/31)**

## Decision

The existing local candidate pool is strongly associated with **HuggingFaceH4/no_robots**, whose upstream data card explicitly specifies **CC BY-NC 4.0**. Local file-derived lineage and commercial reuse rights have NOT been proven. **Treat all candidate held-out source groups as research-only with an explicit noncommercial/licensing gate, and do not fit a real classifier until data-use review, independent receipt custody and semantic near-duplicate acceptance are resolved.**

Authoritative public dataset card: https://huggingface.co/datasets/HuggingFaceH4/no_robots . It states 9,500 training rows and CC BY-NC 4.0. This is public evidence about the upstream release, not a verified cryptographic local-to-upstream file match or an opinion on legal permission for a specific use. No source content was downloaded or uploaded.

## Physical Xwing provenance audit (read-only CPU)

Inspected the existing pinned `general-norobots.jsonl` metadata while keeping all user prompts and response content local. The file has **9,500 conversations** and all 9,500 have known `norobots/<task>` source labels. Category counts:

| Local public-source category | Rows |
| --- | ---: |
| Generation | 4,346 |
| Open QA | 1,182 |
| Brainstorm | 1,060 |
| Chat | 796 |
| Rewrite | 625 |
| Summarize | 395 |
| Classify | 334 |
| Coding | 334 |
| Closed QA | 245 |
| Extract | 183 |
| **Total** | **9,500** |

There is **no license/rights field in the local conversation rows**, so local-to-upstream transformations, licensing provenance and commercial use rights remain unverified. The new `k2_norobots_provenance.py` only emits the upstream-license candidate if **all 9,500 rows match the expected `norobots/` category labels and count**. A shorter synthetic file or inconsistent metadata yields `UNVERIFIED`, never an automatic grant.

**Local source SHA256:** `27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5`.

**Private provenance report:** Xwing `/media/scott/data/finetune-staging/research-witness-20261009/no-robots-provenance-strict-20261009.json`; mode `0600`, SHA256 `e76c8bd9420aaddd1beb762700351f41934bbe08c6d6f17b0edfad7b7a6ff04c`. No raw prompts, responses, secret HMACs or token data were uploaded to GitHub.

## Physical Xwing order-insensitive lexical proxy screen

The previous [PR #31](https://github.com/scottjoyner/auto-finetune/pull/31) exhaustive `SequenceMatcher >= 0.85` audit found **zero matches** between all 210 exact-distinct source prompts from `train.combined.jsonl` and all 10,912 exact-distinct prompts from `general-norobots`. The new `k2_token_set_overlap.py` adds a different, order-insensitive diagnostic: normalized Unicode word **set Jaccard >=0.75**, requiring at least four unique tokens per prompt. All 210 x 10,912 pairs are either scored or proven ineligible under the set-size bound; no approximation silently drops qualifying Jaccard matches.

| Actual metric | Count |
| --- | ---: |
| Exact-distinct original K2 training prompts | **210** |
| Exact-distinct auxiliary prompts | **10,912** |
| Total source-auxiliary comparisons accounted | **2,291,520** |
| Size/short-prompt-pruned pairs | 1,940,899 |
| Token-Jaccard-scored pairs | **350,621** |
| Matches with token Jaccard >=0.75 | **0** |
| Original prompts shorter than 4 unique words | 18 |
| Auxiliary prompts shorter than 4 unique words | 75 |

**Private token audit:** Xwing `/media/scott/data/finetune-staging/research-witness-20261009/no-robots-token-overlap-20261009.json`; mode `0600`; SHA256 `a79a063972131274560e1d8741f1d48ec76e26bc72ec4b01fb4bed7629a52169`.

**Interpretation:** This provides an additional, independent *syntactic* screen for reordered/overlapping words. **It does NOT detect paraphrases with synonyms or distinct terminology**, infer task-level independence, approve arbitrary held-out splits, or establish commercial reuse rights. Short prompts are not meaningfully screened by this token Jaccard rule. There is no local embedding model, human semantic-review acceptance, or independently signed source audit in this slice.

## What remains valid from the prior source-family acceptance

[PR #31](https://github.com/scottjoyner/auto-finetune/pull/31) measured **4,266 K2-tokenizable prompt/response rows** and **4,142 exact-distinct normalized prompts** in the candidate auxiliary pool, collapsing to **4,019 exhaustive lexical families** at character similarity 0.85. Family-level candidate allocation **2,415 train / 768 validation / 836 test** is *not an approved split*. Independent local SHA reconciliation passed. The existing real v2 16-feature K2 dataset still contains ONE old-source episode, not 4,019 trained examples.

## Strict remaining HOLDs

1. **Data rights:** upstream No Robots is marked CC BY-NC 4.0. The local transformation chain has not been verified, and noncommercial restrictions require careful review before any use, particularly in commercial production or training. No local or upstream source text is redistributed.
2. **Semantic source independence:** 0 lexical and token-set matches is not a semantic/paraphrase certificate; need a separate bounded embedding or human-reviewed paraphrase audit with documented false negatives, and source-family linkage before trusted train/val/test freezing.
3. **Receiver signing key:** legacy x1 HMAC credentials were readable via the Xwing-accessible `scott` principal. [PR #27](https://github.com/scottjoyner/auto-finetune/pull/27) proposes a distinct `dustreceipt` UID, but that service has not been provisioned/independently verified. A read-only x1 service-account probe in this session was denied by Fleet Commander's safety controls; no privileged changes occurred.
4. **Real classifier quality:** no genuine new 16D source episodes were collected from this dataset, no weights trained and no claim of improvement over curvature-aware momentum made. Training, biased gradient selection, LoRA checkpoints, NAS writes and production authority remain DENY.

## Reproduce the research-only check

Both CLIs require an explicit read-only switch and new output paths in an existing mode-700 local SSD directory:

```bash
python -m experiments.dust.k2_norobots_provenance \
  --read-only-provenance \
  --local-corpus /media/scott/data/finetune-staging/data/datasets/general-norobots.jsonl \
  --expected-source-sha256 27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5 \
  --private-output /path/to/NEW-private-provenance.json

python -m experiments.dust.k2_token_set_overlap \
  --read-only-token-triage \
  --source-jsonl /media/scott/data/finetune-staging/data/datasets/train.combined.jsonl \
  --auxiliary-jsonl /media/scott/data/finetune-staging/data/datasets/general-norobots.jsonl \
  --expected-source-sha256 2b7b01b0388474af9255c62de0be28aaedb26af2494c25f3bc73493d050e8426 \
  --expected-auxiliary-sha256 27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5 \
  --private-output /path/to/NEW-private-token-audit.json
```

Neither command executes the K2 model, contacts a hosted provider or writes to NAS. Run new checks only in isolated worktrees and never overwrite pre-existing private research evidence.
