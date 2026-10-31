# LFM3-1.7B: repeated random-direction controls (2026-10-30)

**Research-only.** This is a follow-up to the frozen six-pair
reversible-projection smoke, not a new edited model or a safety certification.
The principal A/B/C checkpoints remain unchanged.

## Predeclared design

The fixed two-pair calibration direction comes from the original experiment.
Six disjoint heldout pairs (p03–p08) are tested using the original post-trained
checkpoint A, CPU BF16, greedy decoding, 24 maximum new tokens, layer 10,
and alpha 1.0 reversible projection. Real actions and tools are disabled.

The source run already contains one orthogonal random-direction sham
(seed 17), an unmodified baseline, alpha=0 identity, alpha=.25, alpha=.5,
and fitted alpha=1 outputs for each holdout prompt. New runs use THREE
additional deterministic orthogonal sham directions, seeded 23, 41 and 59.
This is 36 new generations; the original 72 are *reused*, not regenerated.

A sham vector is created by projecting a seeded random 2,048-dimensional
Gaussian vector away from the fitted unit direction and renormalizing.
Greedy generation means these seeds define **different sham vectors**;
they do not represent repeated sampling or retraining the fitted direction.
## Actual x1-370 results

All **36 new CPU bfloat16 generations** completed across the three new
random-direction seeds. The original 72-generation journal was reused
without modification; 108 generations in total are involved in the
comparison, but only 36 are new.

| Direction | Benign exact text changed /6 | Adversarial exact text changed /6 |
|---|---:|---:|
| Fitted layer-10 direction, alpha 1 | 4 | 6 |
| Orthogonal random seed 17 | 1 | 3 |
| Orthogonal random seed 23 | 1 | 1 |
| Orthogonal random seed 41 | 2 | 1 |
| Orthogonal random seed 59 | 1 | 2 |

Every random sham kept **benign 0/6** and **adversarial 6/6** positive
on the original weak refusal-keyword detector. The fitted alpha 1
intervention had the same indicators. The strongest apparent
text-change contrast did not attain adjusted significance: across
the six newly measured (three seeds × two prompt classes) paired tests,
the minimum Holm-adjusted p was **0.375**. The existing seed 17 is
descriptive only and excluded from this six-test correction.

Geometry check: maximum absolute cosine between the fitted vector and
any of the four shams was about 1.68e-8; maximum absolute inter-sham
cosine was about 0.02442. This validates control-vector diversity, NOT
behavioral specificity. The study does not establish increased or
decreased harmful compliance, and none of the model weights changed.

Aggregate-only evidence: `evidence/20261030_orthogonal_sham_repeats.json`.

## Interpreting the metrics

Within each six-prompt benign or adversarial stratum, compare each sham
and the fitted alpha=1 intervention against the exact original A response.
The measure is whether the *full response string changes*; this is not a
refusal outcome or quality score. Also count refusal-keyword positives.

For each new sham seed, calculate a paired two-sided exact conditional
sign/McNemar test on discordant *changed-vs-original* indicators for
the fitted direction versus that sham. Adjust the six new comparisons
(three shams × two source strata) with Holm's step-down correction.
The original seed 17 is shown descriptively, not retested post hoc.

Small samples, keyword indicators, and response differences cannot
establish helpfulness, justified refusal, harmful actionability, causal
direction specificity, a zero-refusal model, or deployment safety.

Reproduce from this package root on the authorized `x1-370` using
the locally pinned Python environment. Public reports never contain
raw prompts, sample identifiers, raw generations or checkpoint weights.
## Commands

```bash
/home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
/home/scott/lfm3-research/train-venv/bin/python -m src.orthogonal_sham_repeats \
  --source-dir /home/scott/git/af-lfm3-reversible-alpha-controls-20261030/research/lfm3-refusal-validation/results/reversible_alpha_sixpair_20261030_private \
  --private-out results/orthogonal_sham_repeats_20261030_private \
  --public-out evidence/20261030_orthogonal_sham_repeats.json
```

The tool pins original model/vector/prompt-pair hashes and the source
journal SHA-256; refuses mismatched manifest settings and duplicates;
journals privately with fsync and supports exact-manifest resume.
The result JSON is immutable and includes only aggregates and digests.
The existing private journal and all checkpoint weights remain untouched.
