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
scores. The attempted model-scoring command did NOT execute.

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
