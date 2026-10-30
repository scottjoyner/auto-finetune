# Exact paired XSTest lexical-proxy significance audit

Status: completed from immutable existing 450-prompt x 3-response journal.
This is **not human refusal grading** or a model safety result.

The matched benign A/B comparison (250 prompts) has 9 refusal-keyword
positives that disappeared under B, 1 newly positive response under B,
30 keyword-positive in both and 210 keyword-negative in both. Net
B-minus-A is minus 8/250, or -3.2 percentage points.

An exploratory two-sided exact conditional McNemar/binomial test gives
p=0.021484375 *without correction*. Across the six exploratory tests
(three arm contrasts in both benign and unsafe strata), the
Holm-Bonferroni corrected p is **0.12890625** for this comparison:
not below 0.05 on the familywise-adjusted criterion. The paired
2,000-resample bootstrap interval previously recorded for the benign
lexical proxy is [-6.0, -0.8] percentage points; this exploratory
interval is not simultaneous across six contrasts. The unsafe A/B
lexical proxy difference is -2.5 percentage points, and it should not
be treated as a desirable or safe reduction in refusal.

The published report checks source/checkpoint/journal hashes against
the previous complete private XSTest audit and checks the binary
paired 2x2 cells against all three model marginals. It computes
aggregate numbers only; no new inference, model modifications,
human annotations, response texts, or unsafe prompt materials.

Re-run the read-only calculation with a NEW output name:

    python -m src.xstest_proxy_exact_paired \
      --aggregate evidence/20261029_full_xstest_proxy_diagnostics.json \
      --results-dir results/xstest_full_20261029_v1 \
      --output evidence/local_new_xstest_paired_exact.json

Evidence: evidence/20261030_xstest_proxy_paired_exact.json

Interpretation: these results do NOT establish actual refusal,
helpfulness, harmful actionability, global zero refusal, or
capability retention. Only two independent complete human graders
and an independent standardized capability benchmark can
address the remaining study endpoints. The original A/B/C lineup
remains frozen and unapproved for release.
