# Joint LFM3-1.7B abliteration tradeoff audit — November 1, 2026

**Research-only.** This is a comparative validation of frozen checkpoints A (original post-trained), B (single M1 projection), and C (pretrained base). No new checkpoint, weights, inference, external transmission of private prompts, or human ratings are created.

## Question and benchmark custody

Does the observed lower refusal-keyword signal in checkpoint B come with preserved general capability? The two **different** independent benchmark families provide a limited, quantitative screen, not a behavioral safety result:

- Original frozen XSTest: 450 matched prompts, 1,350 existing generations, 250 benign and 200 unsafe; matched keyword proxy counts and SHA-256 custody.
- Canonical EleutherAI `arc_challenge`: 1,172 matched multiple-choice questions for each of A/B/C, built-in harness v0.4.13, source content SHA-256 `de86c368d3a59701a5114595cefc1869edac18bd558710f930bf6b6a7a753224`, previously independently checked source and case-level local/canonical parity. These are canonical evaluation results, not the earlier custom ARC adapter.
- All source, checkpoint, paired-count, metric, exact conditional McNemar p-value, original Holm-adjustment, and release-closed invariants are verified before producing the joint report.

## Observed A versus B (right minus left)

| Measurement | Fixed denominator | Paired indicator gains/losses | Observed difference | 95% simultaneous conservative confidence envelope |
| --- | ---: | ---: | ---: | ---: |
| Benign refusal keyword | 250 | 1 / 9 | -3.20 pp | -8.7931 to +2.6177 pp |
| Unsafe refusal keyword | 200 | 5 / 10 | -2.50 pp | -11.2389 to +6.4569 pp |
| Canonical ARC `acc` | 1,172 | 18 / 29 | -0.9386 pp | -3.5452 to +1.6838 pp |
| Canonical ARC `acc_norm` | 1,172 | 14 / 23 | -0.7679 pp | -3.1023 to +1.5808 pp |

Exact intervals are computed using 24 marginal Clopper-Pearson binomial confidence intervals at alpha 0.05/24, combined into conservative difference envelopes covering a joint family of **12 planned matched comparisons (six XSTest lexical, six canonical ARC)**. This union-bound method does not require independent *comparisons*; inference still assumes the benchmark prompts are representative independent draws.

Both capability intervals cross zero and extend below **−2.0 pp**, so the analysis does not justify a claim of capability noninferiority within an illustrative 2-percentage-point loss margin. The intervals are above −5 pp, but a 5-point margin is **post-hoc only** and not pre-registered or accepted as a preservation criterion. Neither keyword difference establishes a behavioral improvement. In particular, the unsafe-keyword reduction could reflect lost appropriate refusals.

C is pretrained rather than instruction-tuned; large C versus A/B score differences are not attributable solely to B's ablation procedure. The alpha=1 fitted direction in B was derived from only two calibration pairs, leaving substantial external-validity uncertainty.

## Reproducible implementation

- `src/joint_abliteration_tradeoff.py` — fail-closed identity, custody, paired-score and uncertainty audit.
- `tests/test_joint_abliteration_tradeoff.py` — synthetic tampering tests, source-contract checks, and frozen A/B observed-score and exploratory-margin assertions.
- `evidence/20261101_joint_abliteration_tradeoff.json` — aggregate-only table, SHA-256 custody receipts, six+six comparisons and non-release assertions.
- Focused x1-370 validation before publication: **16/16 tests passed** in an isolated SSD scratch tree. Hosted CI is tracked separately; a local pass does not imply hosted success.

```bash
cd research/lfm3-refusal-validation
python -m pytest -o addopts= -q tests/test_joint_abliteration_tradeoff.py
python -m src.joint_abliteration_tradeoff \
  --lexical evidence/20261030_xstest_proxy_paired_exact.json \
  --canonical evidence/20261030_arc_canonical_full1172_cpu_aggregate.json
```

The command writes no evidence file unless `--output` is supplied, and refuses overwrites.

## Required next experiments and release gates

Keep the A/B/C checkpoint weights and frozen journals immutable. Collect independent blinded human judgments on *benign useful answers, appropriate refusal, and harmful actionability* from the existing review packets; do not fabricate reviewer scores or interpret keyword matches as behavioral labels. Only after human validation and an independently selected noninferiority margin should a new direction, additional calibration prompts, layer candidates, or strength sweep be proposed as a separate controlled research branch. Preserve a held-out set and matched sham directions; rerun canonical ARC and refusal benchmarks before any research release.

The `x1-370` root filesystem has only ~20.74 GiB ordinary-user free, close to the protective 20 GiB gate. Storage incident `auto-finetune#106` remains open. No large model load, job termination, model promotion, or checkpoint publication was performed.
