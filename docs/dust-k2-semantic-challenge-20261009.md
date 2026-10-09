# Dust/K2 R2.8 — offline semantic challenge and candidate review

2026-10-09 | Research-only draft | stacked on PR #32.

## Scientific decision

The first true embedding-based source audit has completed using locally cached frozen encoders, with no training or external inference. **Both encoders failed the preregistered positive/hard-negative challenge**, although for different reasons.

The frozen challenge consists of 10 same-intent paraphrases and 10 related-topic but DIFFERENT-intent hard negatives. Thresholds were fixed at cosine 0.75, 0.85 and 0.92 BEFORE the first model run.

| At frozen cosine 0.85 | MiniLM-L12 | Multilingual E5-large |
| --- | ---: | ---: |
| Known paraphrases found | 0/10 | 7/10 |
| Hard negatives incorrectly flagged | 0/10 | 8/10 |
| Frozen challenge gate | FAIL | FAIL |
| Positive outscores matched hard negative | 9/10 | 5/10 |
| Pairwise AUC, 100 positive-vs-negative comparisons | 0.79 | 0.57 |
| Mean matched positive-minus-negative cosine | +0.1672 | +0.00445 |

**The threshold-free ranking diagnostics were introduced after the initial threshold failure, and are exploratory, NOT independent heldout acceptance.** In particular the first MiniLM rank result suggests a future ranking-based human review approach; it does NOT authorize adjusting a threshold on these same 20 examples and declaring success. A new, substantially larger, independently reviewed challenge set is required.

## Actual full-corpus offline MiniLM audit

Read-only CPU inference on Xwing evaluated all 210 distinct original training prompts against the 4,142 previously qualified 128-token auxiliary prompts (from 4,019 lexical families). The model scored all 869,820 cross-source cosine pairs and 8,576,011 unordered auxiliary cosine pairs; no approximate index.

| MiniLM cosine threshold | Cross-lexical-family candidate links | Candidate links across provisional train/val/test splits | Lexical families implicated |
| --- | ---: | ---: | ---: |
| 0.75 | 559 | 323 | 571 |
| 0.85 | 157 | **90** | **209** |
| 0.92 | 34 | 13 | 59 |

There were zero original-to-auxiliary cosine matches at the thresholds above. **Do NOT infer semantic source independence from that zero:** the frozen challenge establishes poor paraphrase recall at these cutoffs. The 90 provisional split-crossing edges are potential human-review candidates, not confirmed paraphrases. Do not subtract 209 families as though the links proved equality, and do not promote provisional train/validation/test splits.

## Local-only evidence and model revisions

- Full MiniLM source/within-corpus audit, Xwing private mode 0600: /media/scott/data/finetune-staging/research-witness-20261009/semantic-minilm-review-20261009.json
  SHA256 757ec7271827360aaeb95d2d723d8f7c0e707f7c21d60bcab5429b74627a9dfc.
- MiniLM frozen-challenge ranking follow-up, private mode 0600: semantic-minilm-ranking-challenge-20261009.json
  SHA256 6ee15c474e89b1f25828cac5bd31e9d257b6d355321c182cc3febb3b0863118d.
- Multilingual E5 frozen challenge with explicit query-prefix and no private source corpus, private mode 0600: semantic-e5-ranking-challenge-20261009.json
  SHA256 07661ed79377072cd2ec100fb1a5155bd6d933f0d731a11f928e18ae9025d7bf.

MiniLM-L12 model safetensors SHA256 d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542. Multilingual E5 large model safetensors SHA256 020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6. Original K2 source SHA256 2b7b01b0388474af9255c62de0be28aaedb26af2494c25f3bc73493d050e8426. Auxiliary No Robots SHA256 27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5. Private HMAC source-family manifest SHA256 69642839c28a2e74c0ea5e523424136ef3ca9451cd102244753aeeea4540d3cc.

Prompts, raw embeddings, tokens, pair-level scores, per-source HMAC identities, and signing keys were never copied to GitHub; only source-independent aggregates and SHA identifiers are recorded.

## Files and safety gates

- experiments/dust/k2_semantic_challenge_contract.py — frozen 10 positive/10 hard-negative synthetic challenges, fixed thresholds, ranking diagnostics, strict no-promotion source/split and classifier gates.
- experiments/dust/k2_offline_semantic_audit.py — local-only AutoModel inference on CPU, 4 threads, bounded 48-item batches; verifies pinned source/model/config/HMAC family manifest and computes dense similarities in memory, writing only private 0600 aggregate evidence. Never invokes the K2 model's forward pass for training.
- experiments/dust/k2_encoder_challenge_compare.py — compares an alternative cached E5 encoder on SYNTHETIC challenge items only, with recorded model identity and explicit query prefix.
- tests/test_dust_k2_semantic_challenge.py — five stdlib-only regression tests, including fake promotion denial.
- .github/workflows/dust-k2-semantic-cpu.yml — focused CPU semantic contract test workflow (no model weights, actual corpora, GPUs, provider calls or keys).

## Follow-on acceptance conditions

1. SOURCE RIGHTS: local No Robots rows strongly indicate the public HuggingFaceH4/no_robots dataset, upstream CC BY-NC 4.0. Exact transformation lineage and license use restrictions remain to be approved. Commercial reuse has not been authorized.
2. SEMANTIC CHECK: the TWO encoder/cutoff combinations FAIL on frozen challenge. Author a fresh >32-positive/32-hard-negative test set with topic-matched examples and adjudication. Run independent confirmation with measured false negatives/positives; then prioritize a small private human-review queue of the candidate 90 cross-split embedding links. No positive-vs-negative challenge example may be used to claim independent post-hoc generalization.
3. INDEPENDENT RECEIVER CUSTODY: Xwing can read legacy x1 signing key via shared scott SSH identity. Separate restricted receiver UID service in PR #27 is not provisioned or verified; no old receipt may be upgraded.
4. CLASSIFIER QUALITY: one original real v2 16D K2 episode, and no 64 train/16 val/32 untouched test source groups. The toy selector previously lost to curvature-aware momentum. NO model training, learned direction selection, optimizer writes, NAS writes or production execution.

**Verdict: genuine local semantic triage evidence PRODUCED; semantic detector acceptance FAIL; source/split rights and trusted signer HOLD; classifier and production DENY.**
