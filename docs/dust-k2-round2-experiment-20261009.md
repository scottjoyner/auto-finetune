# Dust/K2 predictive multi-direction optimization — round-two experiment preregistration

**Frozen 2026-10-09 | Research-only design/CPU implementation | Built on [source-cohort PR #24](https://github.com/scottjoyner/auto-finetune/pull/24), [real-model witness PR #23](https://github.com/scottjoyner/auto-finetune/pull/23), and [shadow classifier PR #22](https://github.com/scottjoyner/auto-finetune/pull/22)**

## Why this round exists

The original three-head synthetic utility classifier was **inferior to curvature-aware momentum on its independent confirmation cohort**; it cannot be promoted. Subsequent real K2 ROCm observation created **eight independent source-cluster episodes with 64 complete antithetic labels** and showed producer/x1 ledger consistency. That alone does not provide an adequately sized held-out classifier dataset, comparable real-world feature schema, source paraphrase independence, or receiver signing-key isolation.

The most consequential security observation is **Xwing's `scott` SSH principal can read the supposedly x1-receiver-owned HMAC key through the same `scott` account**. Thus existing signatures are genuine *internal ledger-consistency evidence*, not an adversary-resistant independent witness. Do **not** authorize real-label classifier training based on those signatures.

This slice creates two experimentally useful, self-contained pieces, both NOT wired into live K2 updates:

1. `experiments/dust/k2_cross_corpus_audit.py`: exact pinned-corpus integrity plus bounded cross-corpus normalized-prompt screening at lexical similarity >=0.85; propagates any external overlap to **the entire original near-duplicate cluster** and quarantines its sample indices. Has exact manifest/source-model pin checks, ≤3 auxiliary corpora/≤512 pairs screened per corpus, explicit truncation and strict no-prompt-output behavior. This audit **cannot establish semantic/paraphrase independence**, exhaustive auxiliary coverage if truncated, or classifier readiness.
2. `experiments/dust/k2_causal_feature_history.py`: v2 eight-dimensional **previously completed probe only** features for candidate preview, including number of prior completed directions, prior beneficial-rate, preceding plus-direction gain mean/std, preceding antithetic slope mean/std, preceding absolute slope mean and preceding most recent gain. The component refuses committing outcomes before that direction's pre-probe preview, rejects out-of-order updates, and resets state per masked-prompt episode. This is **not a trained predictor** or an active K2 observer hook. All current v1 witness outputs remain v1; do not silently relabel historical samples as v2.

### Source-corpus audit execution caveat

The code and synthetic test can be validated without access to private corpora. A live scan of additional Xwing corpora was attempted in this session but the command was **blocked by tool safety checks**; no auxiliary-corpus overlap result is claimed. Existing [PR #24](https://github.com/scottjoyner/auto-finetune/pull/24) single-corpus results remain: 75 tokenizable examples at 128 tokens, 62 lexical near-duplicate clusters total, in the first runner window 28 eligible train/11 validation/13 test groups and one cross-partition cluster quarantined. **Those numbers DO NOT include cross-corpus or paraphrase screening**. Never interpret an unrun audit as zero overlaps.

### Frozen round-two experiment plan — per-stage gates

**R2-G0: Credential isolation, HOLD until operator-owned privileged deployment.** Provision a *distinct x1 receiver service Unix UID*, distinct from the Xwing-accessible `scott` principal. The service owns its 0700 directory and new 0600 HMAC key. Expose a **restricted receipt-only** forced-command SSH entry or a tightly bound Unix/service endpoint; disallow general shell, arbitrary file access, port forwarding and sudo for the producer identity. Keep existing fleet admin/SSH controls unchanged pending review. Test *from the actual Xwing producer SSH credentials*: `id -u` maps to a non-owner of the key and read-access attempt returns denied; prove receiver process key access still works; prove both HMAC receipt/label join and refusal on tampering. Rotate the new key and mark all old receipts as `legacy-shared-principal`; **old signatures cannot be promoted retroactively**. No actual receiver-service account was provisioned in this session; no elevated permissions were assumed.

**R2-G1: Corpus independence, HOLD.** Screen each explicitly approved auxiliary local corpus; SHA-pin every file and compare source prompts using exact normalized matches and at least the preregistered 0.85 lexical threshold. For any cross-corpus match quarantine the full existing source cluster. If any auxiliary scan hits its cap, or corpus is unavailable or unapproved, keep `INCOMPLETE_CROSS_CORPUS_SCREEN`. Separately perform manually approved embedding/semantic and near-paraphrase analysis where available, documenting false positives/negatives; lexical-only negative result is never a semantic pass. Do not put prompt text or unkeyed prompt digests in GitHub evidence.

**R2-G2: Sufficient source-level sample count, HOLD.** Choose at least **64 source-disjoint training clusters + 16 validation clusters + 32 untouched test clusters**, plus audit-only duplicates/quarantines. This is **112 independent source clusters**, **not 112 directions**. If the currently available corpora cannot supply these, record `INSUFFICIENT_SOURCE_EPISODES` and stop: do not retry with 32 directions from 4 sources to fake sample size. With initial K=8 complete antithetic directions per episode, minimum 896 direction records. After the isolated credential receiver exists, a new eight-source K=8 batch may test the new v2 recorder and isolation but is **not** enough to meet the classifier gate. Reconfirm data rights/retention first. Existing eight-episode v1 samples are engineering smoke data only.

**R2-G3: Versioned real v2 feature generation, NOT_RUN.** The frozen eight causal-history components from `CausalProbeHistory` are candidate-independent unless prior observed gain history changes; the model must also include candidate-specific pre-loss projections/activation geometry in a **separately versioned and tested schema**. Never present the eight history components alone as a fully discriminative direction classifier feature. For the initial batch of an episode, history is all zeros by design; all directions of a batch share the prior completed batch snapshot. PRE must be committed and optionally receiver-witnessed **before** each plus/minus scoring forward. Plus/minus/current candidate outcomes are forbidden as features. Version/normalize within training source groups only, and do not transfer the prior toy-synthetic logistic weights to new real-feature dimensions.

**R2-G4: Classification benchmark, NOT_RUN.** Only after G0-G3 acceptance fit an *auxiliary, pretrained-free* logistic ensemble with seeds 7/42/1337 against complete direction labels (`clean_CE - plus_CE > 0`), train on source-disjoint groups, select threshold and model on validation groups only, then assess once on the untouched >=32 source-group test cohort. Compare **random, plain momentum, curvature-aware momentum and the learned selector** with identical evaluated candidates and K=8 compute, reporting precision@K, mean true gain@K, Brier/calibration, uncertainty/disagreement, wall-clock overhead and 1000-draw paired bootstrap CIs over **source clusters**, never over directions. No claim of superiority unless precision and gain lower 95% paired limits are both >0 against the best prespecified baseline. Report negative outcomes, class collapse and source-level uncertainty.

**R2-G5: Held-out K2 quality and sampling-bias control, NOT_RUN.** Even if G4 passes, do not replace shared-orthogonal K=1024, sigma=0.25, D4. Any learned selection changes the sampling distribution, so compare explicitly biased selection against unbiased/matched-cost or importance-weighted controls, and demonstrate separately pinned held-out language-model CE/tool-output quality over matched 8/16/32 update schedules. Production inference, NAS writes, hosted provider dispatch and optimizer changes remain DENY.

## Acceptance ledger as of October 9

| Gate | Status | Evidence |
| --- | --- | --- |
| Existing 8-cluster, 64-direction Xwing pilot | **PASS consistency-only** | PR #24 producer ledger/receiver HMAC; models unchanged |
| Independent x1 signing-key custody | **HOLD** | Producer can read receiver HMAC key via shared SSH identity |
| Cross-corpus leakage | **NOT_RUN/BLOCKED** | Implementation exists; live auxiliary-data command blocked |
| Versioned causal-history CPU semantics | **CPU tests implemented** | PRE-before-POST, no future/current candidate outcome, episode isolation |
| Real v2 feature data collection | **NOT_RUN** | No new live K2 model run in round two |
| >=64 train / 16 val / 32 test source groups | **HOLD** | Only 8 independent source episodes observed |
| Classifier fine-tune and confirmatory quality | **NOT_RUN** | Synthetic classifier failed earlier independent confirmation |
| Production K2 optimizer/admission | **DENY** | No scheduler, sampling, model checkpoint or service changes |

## Next operational decision

The fastest safe advancement is **restricted receiver service identity and an independently verified negative key-read test**, then a privacy-reviewed and fully recorded cross-corpus overlap scan. Only after those gates may the experimental v2 recorder collect more source groups. Do not rely on the old receiver key and do not overwrite any existing Xwing/x1 evidence.

The round-two focused GitHub CI exercises synthetic CPU tests only; it cannot attest key isolation, corpus privacy, GPU inference or held-out classifier quality.
