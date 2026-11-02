# LFM3-1.7B XSTest completion-length confounding audit — 2026-10-30

**Status:** Complete descriptive analysis of the original immutable 450-prompt
XSTest × three-checkpoint journal, **1,350 actual completed generations**.
No new weights, inference, human labels or release approval. Refusal keywords
are *not* behavioral refusal grades and length differences are not safety.

## Actual results: all checkpoints

| Frozen arm | Benign median words | Unsafe median words | Benign <50 characters | Unsafe <50 characters |
|---|---:|---:|---:|---:|
| A original instruction-tuned | 87 | 67 | 5/250 | 5/200 |
| B M1-edited A | 88 | 93 | 0/250 | 0/200 |
| C pretrained base | 84 | 8 | 29/250 | 127/200 |

All 127 of the <50-character unsafe C responses had a match on the fixed
refusal-keyword regex. Among its 73 longer unsafe responses, 36 matched.
Similarly, all 29 short benign C responses matched the regex, while only
5 of 221 longer benign C responses did. A's 5 short responses in each
stratum matched the regex; B produced zero short responses. A short
keyword-positive response is **not necessarily an appropriate refusal**.

The original weak indicator full-XSTest totals remain A 39/250 benign and
172/200 unsafe; B 31/250 and 167/200; C 34/250 and 163/200.
## Within-prompt word-count comparisons

| Paired contrast | Benign median delta in whitespace words | Unsafe median delta in whitespace words |
|---|---:|---:|
| B minus A | 0 | +18.5 |
| C minus A | -3.5 | -45 |
| C minus B | -3 | -61 |

Six post-hoc two-sided exact sign tests were Holm-adjusted as a family
(three matched model pairs × two benchmark strata). Adjusted p-values:
benign A/B .288233, A/C .002128, B/C .00005677;
unsafe A/B 7.22e-15, A/C 1.59e-15, B/C 1.82e-23.
These test the directions of paired length differences, **not safety or
refusal efficacy**. This is exploratory secondary analysis and not a
registered length benchmark.

"Words" means whitespace-delimited strings (`str.split()`); NOT model
tokens. "Short" means strictly less than 50 Python Unicode code points;
this 50-character threshold is an exploratory diagnostic, not a clinical,
linguistic, or model behavior threshold. The 128-token generation cap
does not prove whether a particular completion ended naturally or was
truncated; this retrospective journal lacks authoritative finish reasons.
## Integrity, privacy, and next gates

The audit verifies the original benchmark manifest, original model SHA-256
identities, complete journal SHA-256, 450 distinct case IDs and 1,350
matched arm/case responses, class/type tags and original regex outputs.
The public JSON includes only aggregate counts, paired length statistics,
uncertainty p-values and hashes; it includes **no raw prompts or responses,
no case IDs, no reviewer details, and no checkpoint weights**.

The broader 24-layer LFM3 model architecture is unchanged across A/B/C.
C lacks the post-training of A/B; C-vs-instruction-tuned differences
cannot isolate an ablation effect. The original B refusal direction
used only two calibration pairs, so the editing effect also remains
exploratory. Separate independent human review of the complete XSTest
study remains **0/1,350 per grader**, and all release gates remain closed.

Run from the study package on x1-370 with the local pinned train-venv:

```bash
python -m src.xstest_length_confounding \
  --results-dir /home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1 \
  --out evidence/UNIQUE_NEW_xstest_length_audit.json
python -m pytest -o addopts= -q tests
```

Original immutable evidence:
`evidence/20261030_xstest_length_confounding.json`.
Use a new output filename to re-run; the tool refuses overwrite.
