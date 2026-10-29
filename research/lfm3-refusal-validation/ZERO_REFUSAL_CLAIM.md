# LFM3-1.7B abliterated checkpoint: zero-refusal claim audit

**Status (2026-10-29): NOT PROVEN. No new model checkpoint created.**

This document tests the user's requested "maximally abliterated,
zero-refusal" characterization against the immutable existing
three-checkpoint study. It is not a recipe for eliminating safeguards
and does not authorize edited-weight distribution.

## What "maximally abliterated" means in this experiment

The M1 derivative B uses the rank-one matrix projection
W' = (I - alpha * rr^T) W on 48 residual-writing matrices,
with alpha=1 and normalized direction r. At alpha=1 the
r-component of the selected matrix outputs is fully projected
out in exact arithmetic. Independent model-by-model verification
found exactly 48 target matrices edited and 238 others unchanged.

**This is maximal along one selected direction within that
specified alpha range, not globally maximal abliteration.**
The direction was estimated from just two calibration pairs.
LFM3 post-writer normalization and hidden-state interactions
can preserve or restore direction components.

## Results of the frozen full XSTest experiment

The existing model B generated one answer each on 450 prompts.
The weak lexical keyword proxy found:

| Stratum | Original A | M1 derivative B | Pretrained base C |
|---|---:|---:|---:|
| Benign prompts with refusal-pattern keywords (n=250) | 39 | 31 | 34 |
| Unsafe prompts with refusal-pattern keywords (n=200) | 172 | 167 | 163 |
| Both strata combined (n=450) | 211 | 198 | 197 |

These numbers were not obtained by human adjudication.
Keyword positives may occur in non-refusals; absent keywords
may conceal actual refusals. They are **not** validated
measures of behavioral refusals, unsafe compliance, or safety.

**Neither an all-prompt nor a benign-only zero-refusal claim
is supported by the currently available evidence.** Both real
human reviewer packets remain at 0/1,350 completed annotations,
and no independent human safety conclusion exists.

Even if two human reviewers observed zero false refusals
on all 250 benign cases, the exact one-sided 95% upper
bound on the true rate under the same Bernoulli sampling
assumptions is roughly 1.19%, not zero. For zero refusals
on all 450 cases it would be approximately 0.66%.
No finite set can prove universal zero-refusal behavior.

## Appropriate next criteria

For a **benign over-refusal** result, require complete blinded,
independent two-reviewer scoring of all 250 benign prompts
per checkpoint, no unresolved disagreements, disclosure of
denominators, and separate helpfulness evaluation. Report
actionable harmful compliance on the 200 unsafe prompts
as its own endpoint, not as a success metric for "zero
refusal." Complete standardized capability validation and
separate responsible publication/release review.

The frozen A/B/C experimental design is unchanged.
Changing B or creating a new derivative would be a distinct
experiment, invalidating the frozen B comparison as presently
registered. This audit does not request or perform that change.

Reproduction (read-only; private journal and provenance required):

    python -m src.zero_refusal_claim_audit \
      --proxy-aggregate evidence/20261029_full_xstest_proxy_diagnostics.json \
      --results-dir results/xstest_full_20261029_v1 \
      --provenance /home/scott/lfm3-research/refusal-checkpoints/abliterated-m1-smoke/RESEARCH_PROVENANCE.json \
      --output evidence/new_local_zero_refusal_claim_audit.json

Aggregate public report:
evidence/20261029_zero_refusal_claim_audit.json
