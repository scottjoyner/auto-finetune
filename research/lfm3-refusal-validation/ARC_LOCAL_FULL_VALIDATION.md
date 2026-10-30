# Full locally pinned ARC-Challenge audit for frozen LFM3 A/B/C

The canonical 1,172-question ARC test split can be scored with the unchanged
EleutherAI lm-eval 0.4.13 multiple-choice scoring rules using an isolated
local-parquet adapter when the standard Hub dataset is not cached. **This
adapted-task result must not be called an unmodified official leaderboard
result.** The source SHA256 is frozen and every model checkpoint identity is
verified against its original complete safetensors file.

`src/arc_local_full_report.py` is a read-only, fail-closed consumer for three
**completed** no-limit, zero-shot, CPU bfloat16, batch-size-8 runs. Its inputs
are the original restricted results JSON and per-question sample JSONL for
A, B and C, plus the restricted adapter YAML and immutable source parquet.
The tool rejects incorrect model paths, mixed scoring seeds, changed prompt
format or target choice indexes, modified metric aggregation, missing cases,
reordered or edited source questions, fractional per-case scores, inconsistent
aggregate fractions or standard errors, and incomplete 1,172-question arms.
It computes aggregate-only matched A/B/C two-sided exact conditional McNemar
results for each of acc and acc_norm, and Holm-adjusts the six exploratory
hypothesis tests (two metrics times three contrasts). All raw source questions,
case responses, grade decisions and sample logs remain private.

The tool writes a NEW output filename; no earlier evidence may be overwritten.
Every private result and sample file receives a SHA256 receipt. Only the
aggregate and its independently verified statistics may be published.

This is an engineering provenance gate for an adapted task, not independent
reproduction, a proof of capability equivalence, behavioral refusal removal,
or a release authorization. The separate canonical full-ARC gate, two real
blinded human reviewers, adjudication and safety/release approval remain
independent mandatory requirements.

All three A/B/C runs completed all 1,172 cases each on CPU on October 30,
2026. The per-question source and aggregate reconciliation passed for
every arm; a matched aggregate-only statistical report was generated.
See ARC_LOCAL_FULL_RESULTS_20261030.md and
evidence/20261030_arc_local_full1172_cpu_aggregate.json.
Canonical unmodified Hub-task and human review gates remain closed.
