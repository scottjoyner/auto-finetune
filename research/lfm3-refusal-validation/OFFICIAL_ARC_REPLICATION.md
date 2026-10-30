# Independent ARC-Challenge evaluation contract: frozen A/B/C study

Status (2026-10-30): static task/dataset/model provenance ready.
An official model-inference benchmark has not run.

The experiment still uses exactly three original frozen checkpoints:
A original post-trained LFM3-1.7B; B mechanically verified M1-style
rank-one derivative; C original pretrained LFM3-1.7B-Base.

## Canonical standard task

The isolated EleutherAI lm-eval 0.4.13 environment on x1-370 is
at /home/scott/lfm3-research/lm-eval-arc-venv-20261030.
It was installed separately from training and has not changed weights.

The EleutherAI built-in arc_challenge task inherits arc_easy.yaml,
uses the allenai/ai2_arc ARC-Challenge test split, a zero-shot
Question/Answer multiple-choice prompt, one-space target delimiter,
choice texts and answer-key indexes, and independent acc and acc_norm.
The official task validated locally. Source hashes and full exact
task configuration identities appear in the public aggregate evidence.

The pinned local parquet test corpus contains 1,172 distinct cases:
1,165 four-option, 4 three-option and 3 five-option. The model
identity check verifies exact checkpoint SHA256 for A, B and C.
The public evidence contains no question texts or response data.

Reproduce the STATIC check with the local isolated harness Python:

    python -m src.official_arc_readiness \
      --dataset /home/scott/lfm3-research/refusal-checkpoints/arc-benchmark/ARC-Challenge/test-00000-of-00001.parquet \
      --challenge-yaml /home/scott/lfm3-research/lm-eval-arc-venv-20261030/lib/python3.12/site-packages/lm_eval/tasks/arc/arc_challenge.yaml \
      --easy-yaml /home/scott/lfm3-research/lm-eval-arc-venv-20261030/lib/python3.12/site-packages/lm_eval/tasks/arc/arc_easy.yaml \
      --out evidence/local_new_official_arc_readiness.json

A new output filename is mandatory; old evidence is never overwritten.

## Remaining evidence gates

The runner must independently review and authorize the custom local
model implementation, then run the built-in official arc_challenge
task on each original checkpoint with identical zero-shot
configuration, backend, precision and source revision. It must
record both official acc and acc_norm, case counts, failure modes,
and question-matched contrasts. Case-level records stay private.
No official result can be inferred from earlier custom likelihood
scores. The initial canonical-Hub model-scoring command failed before scoring; a separately documented full local-parquet adapted run has now succeeded.

Two independent human reviewers must still grade the 1,350 private
XSTest response records each, adjudicate disagreement and separate
benign false refusal, helpfulness and harmful actionability. Review
cannot be replaced by automated keyword counts.

The three frozen checkpoints must not be changed for this study.
No model release or publication is approved based on this static
readiness audit.

Sources: EleutherAI lm-evaluation-harness arc_challenge.yaml and
arc_easy.yaml, at https://github.com/EleutherAI/lm-evaluation-harness

## Strict intake of independently generated official results (2026-10-30)

The research branch also contains `src/official_arc_result_gate.py`.
After three real *independent* `lm-eval==0.4.13` runs have completed,
the custodian can pass their private results JSON files to this
read-only intake tool alongside the existing static readiness record.
It checks original local model path identity and SHA-256 provenance;
that the task is exactly `arc_challenge` with zero shots, no `limit`,
and all 1,172 questions; and that both official accuracy scores and
their standard errors are present and finite. It will reject an
incomplete/changed arm rather than silently admitting it to analysis.

The gate **does not execute any model**, authenticate the external
compute session, verify per-case question-level outcomes, compute paired
significance, or authorize a release. A JSON file is self-reported;
separate execution receipts, dataset identity attestations, and
independent custody review remain mandatory. All test fixtures are
synthetic and are not benchmark scores. The three actual official
results have not yet been produced.

Examples of required command arguments (replace only with genuine,
locally authorized private result paths): `--static-readiness`,
`--A-original-posttrained`, `--B-M1-abliterated-smoke`,
`--C-base-pretrained`, and `--output` to a *new* private file.
No prompts, raw model completions or private result JSON are committed.

## Successful real 40-question CPU smoke (2026-10-30)

A private local-parquet ARC adapter inherited canonical choice-likelihood scoring because Hub ARC data was absent in the isolated offline cache. The Radeon 890M run failed at the first HIP embedding kernel, but CPU bfloat16 executed all three immutable original checkpoints (40 same ordered questions and 160 choices each). acc / acc_norm: A 20.0 / 25.0%; B 17.5 / 25.0%; C 40.0 / 40.0%. This was **real** model inference, but a small nonrandom locally adapted subset and NOT the full official ARC benchmark. See ARC_LOCAL_CPU_SMOKE_20261030.md and evidence/20261030_arc_local_first40_cpu_aggregate.json. Official complete ARC scoring and independent human review remain pending; model release remains blocked.


## Complete local-parquet ARC test-split scoring (2026-10-30)

Actual CPU inference has NOW completed the entire 1,172-question
test split for each unchanged A/B/C checkpoint. Official-format
choice-scoring rules were inherited through a pinned local parquet
task adapter. Acc/normalized correct counts: A 276/306; B 265/297;
C 459/502. All 3,516 private per-question outputs and the model
and source hashes passed the independent strict audit.
B minus A is -0.94 percentage points raw and -0.77 normalized;
matched exact and six-test Holm-adjusted A/B p=0.287730 for both.
This does not establish equivalence or a real reduction in refusal.
See ARC_LOCAL_FULL_RESULTS_20261030.md and
evidence/20261030_arc_local_full1172_cpu_aggregate.json.

The unmodified canonical Hub task still has NOT completed; this
is the entire test split with *adapted data transport*. Human
review, external replication and release approvals remain blocked.


## Complete canonical built-in ARC-Challenge A/B/C evaluation (2026-10-30)

The unmodified lm-eval v0.4.13 builtin arc_challenge task has NOW
run successfully on the pinned Hugging Face ARC-Challenge
test split, for all 1,172 questions x three unchanged checkpoints.
Exact row-level comparison to the original pinned local parquet
established identical question text, IDs, answer options and
ground truth. All 3,516 per-case binary acc and acc_norm results
also exactly matched the separate local-parquet task results.
Acc/acc_norm percent: A 23.55/26.11, B 22.61/25.34,
C 39.16/42.83. Full private logs were independently audited,
and aggregate-only receipts were published. This closes the
canonical task execution and dataset-equivalence blocker,
NOT the human review, external independent reproduction,
zero-refusal, or release/publication gates. See
CANONICAL_ARC_RESULTS_20261030.md.
