# Dust/K2 R3.0 — Fresh semantic challenge and local-review terminal hardening

**2026-10-09 | Research-only, draft | Stacked on [PR #34](https://github.com/scottjoyner/auto-finetune/pull/34), then #33 and earlier direction-observation PRs.**

## Decisions and results

**Both locally cached frozen semantic encoders FAILED a NEW precommitted 40-positive / 40 topic-matched hard-negative challenge**. This experiment used completely new authored synthetic questions, no K2 training prompts or private No Robots source data, and did not reuse any of the 20 initial PR #33 challenge questions. The authored labels are **not separately human-adjudicated**, so the result is better understood as a disconfirming synthetic stress test than a fully independent human benchmark.

Exact challenge dataset committed *before either model run* at SHA **`0637a7ea40511ce4eed60eb43880cb8de87ae53d`**. Canonical challenge-case digest SHA256:

`20609409c5540e1f1b7facba75038ed2ee37bc8aa8e8e252ed9cce0fe7ff5419`

Forty topic groups each contain an anchor, same-intent paraphrase and topically related but *different-intent* hard negative. Thus 80 scored pairs (40 positive + 40 negative), not 80 statistically independent topics. No text from the private original or No Robots corpora was used. The prespecified acceptance at a single fixed threshold requires **at least 32/40 paraphrase true positives and no more than 8/40 hard-negative false positives**. Thresholds `0.75`, `0.85`, `0.92` were frozen before inference; we do not retune on this set and relabel it untouched confirmation.

| Prespecified threshold / diagnostic | MiniLM-L12 | Multilingual E5-large |
| --- | ---: | ---: |
| Cosine 0.75: paraphrase recall | 9 / 40 | 40 / 40 |
| Cosine 0.75: false-positive hard negatives | 2 / 40 | 40 / 40 |
| **Cosine 0.85: paraphrase recall** | **0 / 40** | **36 / 40** |
| **Cosine 0.85: false-positive hard negatives** | **0 / 40** | **30 / 40** |
| Cosine 0.92: paraphrase recall | 0 / 40 | 8 / 40 |
| Cosine 0.92: false-positive hard negatives | 0 / 40 | 5 / 40 |
| Matched-topic paraphrase outranks negative | 30 / 40 | 25 / 40 |
| All-positives vs all-negatives pairwise AUC | 0.686875 | 0.619375 |
| Mean matched-topic cosine gap | +0.101395 | +0.014947 |
| **Frozen challenge acceptance** | **FAIL** | **FAIL** |

The original PR #33 10-positive/10-negative challenge also failed for both encoders. MiniLM's ranking has some utility for *generating leads*, but 0% recall at cosine 0.85 makes any "zero detected overlapping source prompts" conclusion from its cross-corpus scan untrustworthy as a proof of semantic independence. E5 catches most paraphrases at 0.85 but flags 75% of different-intent hard negatives. **Neither should approve test/train split release, automatic quarantine, classifier learning or changes to the stochastic gradient estimator**. A genuine next scientific gate requires at least two independently human-reviewed annotations of a fresh challenge, with untouched calibration/confirmation separation and confidence intervals grouped by topic.

## Real local Xwing evidence

Both checks ran under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` with preexisting CPU PyTorch/Transformers weights on Xwing and a fixed four-thread ceiling. They did not perform a K2 model forward pass, backpropagation, optimizer update or provider request. Challenge prompts and code are synthetic and public; individual model cosine scores remain ephemeral and only aggregate statistics were retained in private mode-600 report files:

- MiniLM model safetensors SHA256 `d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542`. Report Xwing `/media/scott/data/finetune-staging/research-witness-20261009/fresh40-minilm-challenge-20261009.json`; file SHA256 `1d16482f81c8dd8013419a8f7eb3b54641d5901b356d541803ae59893b7d7e83`; file mode 600, 1,721 bytes.
- E5-large model safetensors SHA256 `020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6`, with recorded `query: ` prefix. Report Xwing `/media/scott/data/finetune-staging/research-witness-20261009/fresh40-e5-challenge-20261009.json`; file SHA256 `69c0b567a987832b3710f9f392df6ad7b0896eb909ca8b5c9e4c628fdb4f7033`; file mode 600, 1,730 bytes.

This is not an independently human-labeled evaluation or evidence that No Robots is licensed for commercial reuse.

## PR #34 local reviewer security refinement

The inherited review interface correctly refused non-TTY execution but **TTY alone does not establish local trust**. SSH can allocate a pseudo-terminal, and source prompts may contain ANSI / OSC 52 control sequences, carriage returns, newlines, bidirectional text overrides and other invisible Unicode controls. Displaying those strings unescaped can manipulate terminal state, misrepresent reviewer content or interact with the clipboard.

This PR updates `k2_local_semantic_review_session.py`:

1. **`display_safe_prompt`** escapes control/invisible Unicode codepoints using visible `\\uXXXX` notation, including `ESC`, BEL, line breaks, U+202E direction overrides and zero-width markers; rejects empty/oversized source text beyond 4,096 characters instead of silently truncating a review.
2. **`refuse_remote_reviewer_session`** denies absent TTY and detected `SSH_CONNECTION`, `SSH_CLIENT`, `SSH_TTY` or `MOSH_IP` environment indicators before opening private prompts.
3. No plaintext prompt or terminal output enters the mode-600 review receipt, which still stores only queue digests and human choices. Existing no-training/source-rights conditions remain unchanged.
4. **Limitation:** environment checks cannot cryptographically prove a physical local terminal, prevent all remote-session forwarding, block terminal recording, or authenticate human identity. Self-attested reviewer IDs remain untrusted. Real reviewer permission and independently controlled account proof are separate gates, not implied by these checks.

Eight new focused tests prove challenge immutability, scores/acceptance, ANSI/OSC/bidi escaping, oversized-input rejection, non-TTY and SSH denial, and absence of raw prompts from label-only receipts. The four previous local-review tests and five private-queue/adjudication tests also pass: **17/17 x1 focused tests**. No real private 90-item semantic queue or human-labeled result was created.

## Next acceptance prerequisites

- **Human semantic benchmark:** authorize a privacy-preserving genuinely independent reviewer process, label a fresh challenge and sample both near- and far-neighbor pairs to measure false negatives as well as false positives. Keep calibration and untouched confirmation separate; do not use the authored 40-topic set as independently human-validated evidence.
- **Dataset provenance/rights:** upstream HuggingFaceH4/no_robots is marked CC BY-NC 4.0; local transformation chain and legal permission appropriate to intended commercial use remain unverified. No commercial classifier fit should use it.
- **Signing-key custody:** old x1 receiver HMAC key was readable under Xwing's shared `scott` SSH principal. Separate-UID restricted receipt signer proposed in PR #27 is unprovisioned and negative producer-access proof remains HOLD. Earlier receipts cannot retroactively acquire stronger trust.
- **Real K2 data:** only one actual v2 16D source episode exists, despite 4,019 candidate lexical families. Preregistered minimum 64 train / 16 validation / 32 untouched test source groups, real classifier-vs-curvature baseline, source-group bootstrap and estimator-bias/heldout K2 CE gates remain NOT RUN.

**Disposition: New frozen 40+40 CPU semantic stress test COMPLETE and BOTH models FAIL; reviewer terminal hardening PASS in focused tests; real local human review, signer isolation, rights, classifier training, optimizer and production remain HOLD/DENY.**
