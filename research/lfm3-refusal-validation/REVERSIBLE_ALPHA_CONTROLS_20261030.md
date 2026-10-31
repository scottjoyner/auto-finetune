# LFM3-1.7B reversible M1 alpha controls — October 30, 2026

Research-only; no checkpoint weights modified. The prior frozen A/B/C
benchmark remains the primary comparison. These *reversible inference hooks*
are method controls, **not** additional frozen checkpoints and are **not**
the rank-one output-matrix weight edit used to construct checkpoint B.

## Scope and causal limitations

- Source: original A post-trained model only, CPU bfloat16 and greedy seed 17.
- Fixed layer 10; original measured difference-of-means vector from two
  calibration pairs; independent local holdouts are p03 through p08.
- Conditions: unmodified A; alpha=0 identity hook; alpha=0.25, 0.50, 1.00;
  deterministic random direction perpendicular to the fitted direction at alpha=1.
- Compare all conditions within each same heldout prompt. The orthogonal
  random control tests direction specificity, not a perfect matched placebo.
- Treat benign lexical positives as *possible* over-refusal, unsafe positives
  as *possible* appropriate refusal, and neither as verified behavior.
- Alpha=0 must exactly reproduce the baseline or the reporter fails closed.
## Actual heldout smoke findings

Completed on x1-370: 6 disjoint holdout pairs, 2 prompt types per pair,
6 conditions per prompt = **72 real CPU BF16 generations**, each capped
at 24 new tokens. All alpha=0 outputs exactly matched unmodified A.

| Condition | Benign keyword /6 | Unsafe keyword /6 | Benign full-text changes /6 | Unsafe full-text changes /6 |
|---|---:|---:|---:|---:|
| Original A | 0 | 6 | 0 | 0 |
| Alpha 0 | 0 | 6 | 0 | 0 |
| Alpha 0.25 | 0 | 6 | 1 | 3 |
| Alpha 0.50 | 0 | 6 | 2 | 5 |
| Alpha 1.00 | 0 | 6 | 4 | 6 |
| Random direction alpha 1.00 | 0 | 6 | 1 | 3 |

A smaller one-holdout-pair integration run independently completed
12/12 generations and also passed alpha=0 parity. In the six-pair run,
lexical classifications were identical across all interventions,
but exact response bytes became more often different at stronger
fitted-direction projection. Random-direction controls also changed
some responses; full-text difference is not necessarily meaningful
behavioral change. No safety or improved refusal efficacy is established.

Read-only aggregate evidence (NO raw generations):
`evidence/20261030_reversible_alpha_smoke.json`,
`evidence/20261030_reversible_alpha_smoke_divergence.json`,
`evidence/20261030_reversible_alpha_sixpair.json`,
`evidence/20261030_reversible_alpha_sixpair_divergence.json`.

## Reproduction on x1-370

Use the pinned training environment and the study package at
`research/lfm3-refusal-validation`. A NEW output directory is required
for a different token budget, holdout count, or experiment revision.

```bash
/home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
/home/scott/lfm3-research/train-venv/bin/python -m src.reversible_alpha_controls \
  --pairs 1 --max-new-tokens 24 --layer 10 \
  --results-dir results/reversible_alpha_smoke_20261030_private \
  --public-out evidence/20261030_reversible_alpha_smoke.json
```

The runner verifies SHA-256 identity for the **original model**,
calibration vector, and eight-pair source; rejects orphaned, incomplete,
duplicated, or inconsistent private journals; fsyncs generation records;
and can resume the same manifest without regenerating completed records.
All responses remain in ignored `results/` with restrictive permissions.
Only aggregate lexical counts and provenance digests may enter the repo.

## Interpretation and gates

Any short smoke sample is descriptive only and too small for reliable
effect sizes, capability retention, or safety conclusions. The
independently completed 450-case XSTest A/B/C study still requires two
blinded real human reviewers of 1,350 answers each. A proper confirmation
would use a separately prespecified larger calibration set, disjoint
heldouts, multiple direction seeds, raw-completion controls, and
human-rated benign helpfulness, justified refusal and harmful actionability.

Do not release edited weights or deploy projected inference as a safety
intervention on the basis of keyword probes or this smoke experiment.
