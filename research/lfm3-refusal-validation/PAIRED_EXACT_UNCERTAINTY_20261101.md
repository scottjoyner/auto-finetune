# Frozen LFM3-1.7B paired keyword uncertainty — November 1, 2026

**Scope: research-only, no new LFM3 model load, checkpoint edit, deployment, raw prompt publication, or human safety grading.**

The original A/B/C XSTest comparison has 450 matched prompts and 1,350 completed generations. Weak lexical refusal-keyword counts are used solely to audit whether the observed per-prompt differences are distinguishable from sampling uncertainty. This **does not** measure the appropriateness of a refusal, harmful actionability, or benign task utility.

## Why this increment matters

Prior `paired_abliteration_power.py` describes conditional sample-size estimates under a normal approximation and fixed observed gain/loss rates. When the number of discordant keyword outcomes is small, those point estimates can look overly decisive.

The new `src/paired_exact_uncertainty.py` verifies the frozen three-checkpoint aggregate through the existing `paired_abliteration_power.analyze` hash, count, exact McNemar, and six-test Holm checks. For each of six matched contrasts it treats newly-positive and no-longer-positive keyword counts as two marginal binomial rates. Each receives a two-sided Clopper-Pearson exact marginal interval with alpha = 0.05/(6*2). The difference of these intervals yields a conservative simultaneous 95% familywise envelope across the six correlated comparisons by a union bound. No independence among contrasts is needed for the union bound, but binomial sampling of benchmark prompts is assumed.

| Fixed contrast | Paired cases | Observed B minus A | Conservative simultaneous 95% envelope |
| --- | ---: | ---: | ---: |
| Benign keyword indicator | 250 | -3.2 percentage points | -8.3743 to +2.2114 points |
| Unsafe keyword indicator | 200 | -2.5 percentage points | -10.6485 to +5.8496 points |

**All six matched contrasts' conservative envelopes include zero.** This is not an equivalence result. The envelopes describe uncertainty in a weak keyword proxy, not in verified refusal behavior. The original benign A-vs-B six-test Holm adjusted p-value remains 0.12890625. No promotion of the B checkpoint is supported.

## Reproduction and integrity

- Source: `src/paired_exact_uncertainty.py`.
- Tests: `tests/test_paired_exact_uncertainty.py`; **10/10 passed** using the frozen original aggregate and synthetic boundary/tamper checks on x1-370.
- Aggregate-only report: `evidence/20261101_paired_exact_uncertainty.json`.
- The original source is `evidence/20261030_xstest_proxy_paired_exact.json` and its SHA-256 model/benchmark/journal identities are checked by the shared analyzer.
- No private prompts, completions, XSTest source IDs, journal contents, or reviewer ratings are included in the new report.

To reproduce in the study root with Python 3.12:

```bash
python -m pytest -o addopts= -q tests/test_paired_exact_uncertainty.py
python -m src.paired_exact_uncertainty --aggregate evidence/20261030_xstest_proxy_paired_exact.json
```

The CLI prints aggregate output without creating a file by default. Optional `--output` refuses to overwrite an existing path.

## Next decision gates

Retain the original A/B/C frozen evaluation, require independent blinded human assessments on benign helpfulness, justified refusal and harmful actionability, and confirm model capability using the separate ARC evaluations. Checkpoint C is pretrained, so C-vs-post-trained differences do not isolate an abliteration effect. PR remains draft, and storage incident `auto-finetune#106` remains open; the live node has only a narrow cushion above its protected 20 GiB root minimum.

Exact marginal binomial bounds do not fix possible non-random benchmark selection, two-pair direction calibration, correlated prompts, keyword validity, or response truncation. No claims of zero-refusal behavior or modified-checkpoint release are authorized.
