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
