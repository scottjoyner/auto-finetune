# LFM3 human-reviewed tradeoff gate — 2026-11-01

Status: research-only; reviewer ratings are still pending. The new `src/reviewed_tradeoff_multiplicity.py` never creates or edits model weights, runs inference, or auto-labels responses.

The evaluator requires two fully completed independent blinded XSTest reviewer assignments covering all 1,350 A/B/C responses each. It reuses existing source-custody checks and private batch readers. Missing grades, changed source text, disagreement or mismatched checkpoint mappings block analysis. Third-human adjudication remains a separate existing workflow.

The twelve prespecified contrasts cover three matched checkpoint pairs for each of four human-rated outcomes: benign over-refusal, benign helpfulness, unsafe actionability, and unsafe full refusal. For binary outcomes, paired differences use conservative Clopper-Pearson bounds across the 12-comparison family. Ordinal helpfulness uses a seeded paired bootstrap; its coverage is approximate.

No private prompt, response, blinded ID, reviewer identity or mapping is published. The output stays in an owner-only directory, is created with owner-only file permissions and cannot overwrite earlier results. Synthetic tests exercise arithmetic and rejection logic but are not real human evidence.

On an authorized host, invoke:

```bash
python -m src.reviewed_tradeoff_multiplicity \
  --packet-dir PRIVATE_SOURCE_PACKET \
  --batches-dir PRIVATE_REVIEW_BATCHES \
  --output PRIVATE_0700_DIRECTORY/human_tradeoff.json
```

Model promotion remains blocked. The earlier XSTest lexical changes and canonical ARC scores do not establish an actual improvement in appropriate refusal or benign utility. Storage incident #106 and draft PR #103 remain open. Fleet Commander was unavailable for live host benchmarking during this change.
