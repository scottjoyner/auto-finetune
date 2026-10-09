# Dust/K2 R2.9 — Private cross-split semantic review, not automatic deduplication

2026-10-09 | Research-only draft | stacked on [PR #33](https://github.com/scottjoyner/auto-finetune/pull/33).

## Why this experiment exists

The completed, SHA-pinned PR #33 MiniLM audit on Xwing found 157 candidate cosine>=0.85 links across lexical source families, of which **90** connect opposite provisional train/validation/test partitions. However MiniLM failed the frozen same-intent/hard-negative challenge: it recalled **0/10** true paraphrases at 0.85. A high cosine is a *review lead*, and no hit is NOT a semantic nonduplication certificate.

The remaining task is human adjudication under controlled local access. This branch establishes a real reproducible interface and acceptance tests, not human judgments or data-use rights.

## Code and privacy boundaries

1. **k2_private_semantic_review_queue.py**: requires exact pinned SHA256 for the prior PR #33 MiniLM aggregate, the PR #31 private family manifest, the No Robots local source and K2 config, plus an existing private source-HMAC key. Repeats the frozen MiniLM CPU forward-only embedding inference under the exact same 0.85 threshold (not a new cutoff). Reconstructs unique HMAC-addressed review pairs **only across provisional partitions and existing lexical families**. Before exclusive-creating a mode-600 private queue, requires that the rederived full cross-family and cross-split counts match exactly **157 and 90**, as witnessed in the pinned PR #33 aggregate. No prompts, targets, tokens, embeddings or plaintext IDs in stdout or GitHub.
2. **k2_local_semantic_review_session.py**: only runs with an actual human **TTY** and pinned queue+corpus+key; rejects batch execution. Displays a pair's two source prompts on the person's **local terminal only**, never saves them into a receipt. A human selects SAME_INTENT, DIFFERENT_INTENT, or UNCERTAIN. Invalid/missing decisions abort without writing; the private receipt is SHA-bound to the original queue. Do not use Fleet Commander or any centralized terminal session to display these private prompt pairs. Human review is limited to an operator who has rights to inspect the local corpus.
3. **k2_human_semantic_adjudication.py**: validates two distinct reviewer identifiers and two complete, independently supplied receipt files, each SHA-bound to the same private queue. If both people agree SAME_INTENT it counts one candidate same-intent match; if both agree DIFFERENT_INTENT it counts one nonmatch; any disagreement/uncertainty stays UNRESOLVED. Aggregate output contains counts and evidence hashes, **never prompt HMACs, per-pair scores, raw text or reviewer identifiers**.
4. **Known limitation:** named reviewer IDs and a true/false declaration are **self-attested**, not cryptographically trusted reviewer identities. An actor able to forge both receipts can spoof agreement, even when the file schema validates. This gate is at best local process consistency until independently controlled human accounts and signed receipts are proven. It does NOT validate MiniLM's false-negative rate because the queue contains only pairs flagged by MiniLM. A new independent annotated challenge must test non-neighbor examples too.

**No classifier training, test split release, automatic quarantine or production authorization** occurs at any stage.

## Existing pinned inputs on Xwing

Private corpus: /media/scott/data/finetune-staging/data/datasets/general-norobots.jsonl; SHA256 `27bc670ee69851923720a5e6014444b26a2bc4fc5df57136915a8821392ed2d5`.

Private lexical family manifest: /media/scott/data/finetune-staging/research-witness-20261009/general-norobots-lexical-clusters-20261009.json; SHA256 `69642839c28a2e74c0ea5e523424136ef3ca9451cd102244753aeeea4540d3cc`.

PR #33 aggregate semantic report: /media/scott/data/finetune-staging/research-witness-20261009/semantic-minilm-review-20261009.json; SHA256 `757ec7271827360aaeb95d2d723d8f7c0e707f7c21d60bcab5429b74627a9dfc`.

Frozen K2 config SHA256 `0ba8f6a0fe8daa5003f88c335735cabc7dba20600ace939efab949ae5e59b936`, local pretrained MiniLM-L12 encoder weights SHA256 `d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542`. The source HMAC key must NEVER be copied into GitHub. All data operations are confined to private Xwing SSD, not the bottlenecked NAS.

## Attempted real experiment and honest blocker

A bounded real Xwing invocation to reconstruct the 90 review candidates was **blocked by fleet-command execution safety controls**. We did not retry with a different invocation or bypass the restriction. Thus **no new 90-item private review queue was created in this pass** and no review receipt exists. PR #33's 157/90 counts remain genuine previously witnessed *aggregate* evidence, not a completed private human-review workflow.

At the implementation branch head, **9/9 synthetic standard-library tests PASS on x1**: five review queue + two-reviewer checks and four interactive TTY/receipt/privacy checks. These do not substitute for actual independent human annotation. Dedicated GitHub workflow is `.github/workflows/dust-k2-private-review-cpu.yml`. Broader repository CI has an unrelated existing failure and must not be represented as all green.

## Future approved manual protocol

An authorized local reviewer, after confirming No Robots data rights and host access, can run the documented private queue CLI directly in an approved Xwing terminal, under an identity that can read the private corpus and research key; it will refuse existing output paths, mismatched digests or incomplete replay. The expected counts are **157 total cross-family candidate links** and **90 cross-split review pairs**, and failure must produce NO accepted queue.

Two genuinely separate reviewers each operate in an interactive *local* session with the exact same immutable queue and independently save mode-600 receipts. No direct prompt text, receipts, reviewer IDs or source HMACs should be sent to the central LLM/session or public GitHub. After both reviewers finish, their receipt SHA256s and queue SHA256 can feed `adjudicate_private_queue` to produce an aggregate read-only agreement report. Never auto-apply those judgments to stored source partitions.

### Next fresh semantic challenge experiment (not yet executed)

- Draft a NEW, **pre-registered** set with at least **32 genuine paraphrases and 32 topic-matched different-intent hard negatives**, deliberately not recycling the first 20 challenge examples, any private held-out K2 episodes, or the review queue's high-cosine pairs as a sole evaluation set.
- Require independent annotators to validate intent equivalence before comparing encoder predictions. Include short prompts, syntax near matches, reordered prompts, synonyms and hard negatives with minimal token differences. Preserve a frozen challenge SHA before inference, with complete labels, examples and auditing held privately under approved rights.
- Measure source-family bootstrap uncertainty for recall/false-positive rate and ranking/AUC, using a **fresh untouched** held-out challenge to assess any later calibrated threshold. An evaluation set used to choose a cutoff cannot be called fresh confirmation.
- Do not declare original/No Robots corpus semantic independence merely because neither MiniLM nor E5 detected some cross-source neighbors. Both frozen encoder configurations failed PR #33 acceptance.

## Unchanged HOLD / DENY

- Upstream HuggingFaceH4/no_robots is identified as CC BY-NC 4.0; actual local transformation rights and intended use remain unapproved. No commercial classifier fitting.
- The original x1 receiver HMAC key was accessible to the producer via a shared `scott` SSH identity; restricted separate-UID signing service from PR #27 has **not** met independent negative-access acceptance.
- Actual v2 K2 16D direction evidence still consists of ONE original source episode; >=64 training, 16 validation and 32 independent untouched test groups remain the preregistered minimum.
- Earlier synthetic direction classifier did not beat curvature-aware momentum on its own frozen independent test. No new stochastic direction sampler, model/LoRA weight updates, GPU training, NAS writes, hosted provider use or production deployment.

**Verdict: private review software and tamper-deny CPU tests PASS; live private queue creation BLOCKED; reviewer adjudication NOT RUN; semantic independence/data rights/signer custody HOLD; classifier training and production DENY.**
