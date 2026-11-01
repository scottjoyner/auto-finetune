# LFM3 human-reviewed tradeoff gate — 2026-11-01

Status: research-only; reviewer ratings are still pending. The new `src/reviewed_tradeoff_multiplicity.py` never creates or edits model weights, runs inference, or auto-labels responses.

The evaluator requires two fully completed independent blinded XSTest reviewer assignments covering all 1,350 A/B/C responses each. It reuses existing source-custody checks and private batch readers. An additional before-and-after SHA-256 digest pass covers **all 30 mutable review batches**, including human grade fields, rather than checking completeness counts alone. Missing grades, changed source text or ratings, unresolved disagreement, or mismatched checkpoint mappings block the direct-consensus analysis. The existing private third-human adjudication workflow now embeds the same 12-comparison analysis in its private final report **after** original reviewer disagreements have been fully resolved; it does not auto-assign grades or publish a model.

The twelve prespecified contrasts cover three matched checkpoint pairs for each of four human-rated outcomes: benign over-refusal, benign helpfulness, unsafe actionability, and unsafe full refusal. For binary outcomes, paired differences use conservative Clopper-Pearson bounds across the 12-comparison family. Ordinal helpfulness uses a seeded paired bootstrap; its coverage is approximate.

No private prompt, response, blinded ID, reviewer identity or mapping is published. The output stays in an owner-only directory, is created with owner-only file permissions and cannot overwrite earlier results. Synthetic tests exercise arithmetic and rejection logic but are not real human evidence.

On an authorized host, invoke:

```bash
python -m src.reviewed_tradeoff_multiplicity \
  --packet-dir PRIVATE_SOURCE_PACKET \
  --batches-dir PRIVATE_REVIEW_BATCHES \
  --output PRIVATE_0700_DIRECTORY/human_tradeoff.json
```

To resolve a completed adjudication set (the authorized on-host reviewer must provide actual paths), run the pre-existing `src.reviewer_adjudication` CLI in `finalize` mode. Its owner-only final report now contains a `human_behavioral_familywise_tradeoff` section; that section is derived from the **third-human resolved source ratings**, not from assumed agreement. Do not upload that private result to GitHub.

A custody consistency recheck and before/after grade digests are read-time checks, not an atomic transaction protecting against a hostile concurrent writer. Freeze reviewer input files while producing the final scientific result.

Model promotion remains blocked. The earlier XSTest lexical changes and canonical ARC scores do not establish an actual improvement in appropriate refusal or benign utility. Storage incident #106 and draft PR #103 remain open. Fleet Commander was unavailable for live host benchmarking during this change.
