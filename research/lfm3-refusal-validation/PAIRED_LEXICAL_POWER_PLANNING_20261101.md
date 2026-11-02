# Frozen LFM3-1.7B paired lexical power planning — November 1, 2026

**Research-only; no additional checkpoint, inference, human review, deployment, or safety claim.**

## Scope and provenance

This analysis reads the original aggregate-only, SHA-256-bound A/B/C XSTest paired-keyword evidence from `evidence/20261030_xstest_proxy_paired_exact.json`, representing 450 matched benchmark prompts (250 benign, 200 unsafe) and 1,350 existing generations. The program verifies the original model/source digest shapes, all six paired contingency tables, published exact conditional McNemar p-values, and six-test Holm-Bonferroni corrections before producing a new aggregate report.

- Implementation: `src/paired_abliteration_power.py`.
- Regression tests: `tests/test_paired_abliteration_power.py`.
- Aggregate report: `evidence/20261101_paired_power_planning.json`.
- Standalone focused regression on x1-370: **14/14 passed**, with no model loads.
- The report includes no private prompt, raw completion, case identifier, or reviewer rating.

## Existing matched A/B results

| Contrast | Keyword transitions 0→1 / 1→0 | Net lexical difference | Published unadjusted exact p | Published six-test Holm adjusted p |
| --- | ---: | ---: | ---: | ---: |
| Benign, B vs A | 1 / 9 across 250 | -3.2 percentage points | 0.021484375 | 0.12890625 |
| Unsafe, B vs A | 5 / 10 across 200 | -2.5 percentage points | 0.3017578125 | 1.0 |

Neither A/B comparison meets the predeclared six-test familywise threshold. All observations are **keywords**, not verified refusal or harmfulness judgments. Reduced unsafe refusal-keyword detection is **not necessarily an improvement**.

## Exploratory sensitivity estimates — not sample-size commitments

Assume each future independent matched prompt replicates the observed keyword gain/loss probabilities; model the paired difference using the normal approximation to a paired binary score test. Reserve two-sided alpha=0.05/6 per comparison (Bonferroni design allowance, more conservative than testing only A vs B). Under these strong assumptions, estimated *total* matched cases for ~80% approximate power are:

| Comparison | Benign cases | Unsafe cases |
| --- | ---: | ---: |
| B vs A | 471 | 1,451 |
| C vs A | 3,267 | 745 |
| C vs B | 9,752 | 4,539 |

The estimates are highly sensitive to very sparse discordance, post-hoc chosen effect sizes, and the calibration sample. They do not create experimental evidence, certify a change in refusal behavior, or establish a justified recruitment size. C is pretrained rather than post-trained; its comparison cannot isolate the effect of the B edit.

## Operational and publication gates

The source/report/tests were prepared and validated in an isolated directory on the distinct ext4 SSD. Root storage incident `auto-finetune#106` is still open. No checkpoint weights, original private journals, reviewer labels, active jobs, or production services were modified.

Continue the frozen three-arm A/B/C design. Prefer independent blinded human assessment of benign helpfulness, justified refusal and harmful actionability over stronger interventions justified only by keyword counts. Do not release or promote models on the basis of these exploratory planning numbers.
