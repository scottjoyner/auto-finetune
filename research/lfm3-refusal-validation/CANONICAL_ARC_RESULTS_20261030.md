# Complete built-in EleutherAI ARC-Challenge A/B/C reproduction — 2026-10-30

The canonical built-in EleutherAI lm-eval 0.4.13 task
arc_challenge was ACTUALLY executed on all 1,172 ARC-Challenge
test questions for the THREE unchanged LFM3-1.7B checkpoints,
zero-shot, CPU bfloat16, batch size 8, with private per-case
logs. All three runs exited zero.

The Hugging Face dataset was loaded successfully at the original
pinned repository revision
210d026faf9955653af8916fad021475a3f00453.
Every question text, answer key, choice list and source ID
exactly matched the previously used local parquet, in order,
for all 1,172 cases. SHA256 of canonical sorted-key JSON rows:
de86c368d3a59701a5114595cefc1869edac18bd558710f930bf6b6a7a753224.
Canonical source and built-in task checksums are pinned
in the public aggregate report.

## Actual canonical scores

| Frozen model | Official-format acc | Official-format acc_norm |
|---|---:|---:|
| A original instruction tuned | 276/1172 = 23.55% | 306/1172 = 26.11% |
| B rank-one edited derivative | 265/1172 = 22.61% | 297/1172 = 25.34% |
| C original pretrained base | 459/1172 = 39.16% | 502/1172 = 42.83% |

Each run performed 4,687 model choice-likelihood requests, 14,061
total, on the built-in task. A separate case-level verifier
checked ALL 3,516 canonical source documents and all binary
acc/acc_norm judgments against the pinned parquet and
the reported means/standard errors.

## Independent transport consistency check

The same all-case check compared the three canonical task
results with our previous full 1,172-question local-parquet
adapter runs, on exactly the same frozen A/B/C model files.
**All 3,516 pairs of acc and acc_norm outcomes were identical,
zero case-level mismatches across either metric.**

The earlier adapter was not silently relabeled as canonical:
the original canonical runs were performed separately using
the builtin task and Hugging Face dataset source.

## Matched statistical analysis

For B relative to A:
- Raw acc: 18 newly correct, 29 no longer correct,
  difference -0.9386 percentage points, exact two-sided
  conditional McNemar p=0.143865, six-comparison Holm p=0.287730.
- Normalized acc: 14 newly correct, 23 no longer correct,
  difference -0.7679 percentage points, nominal p=0.187742,
  six-comparison Holm p=0.287730.

The A/B changes are not significant at 0.05 after correction,
but **lack of significance does not prove equivalent capabilities,
non-inferiority, or preserved instruction following**. Checkpoint C
was not instruction-tuned like A/B and cannot be used as an
unconfounded edit-effect control.

## Provenance and evidence custody

Public aggregate-only report:
evidence/20261030_arc_canonical_full1172_cpu_aggregate.json.

It includes canonical task/parent YAML hashes, pinned dataset
content hash, complete frozen model SHA256 identities, six
paired matched exact tests and Holm-adjusted results,
per-arm means and standard errors, private result/sample
SHA256 receipts, and an explicit zero-mismatch comparison
against the earlier adapted task case-level judgments.

Raw canonical result JSON, individual sample JSONL, task
outputs, prompts and response texts remain private under
results/lm_eval_arc_canonical_private on x1-370. Raw previous
local-adapter JSON/JSONL also remain private. None were
committed to public GitHub. Model weights were not changed.

**Remaining mandatory gates:** two independent human reviewers
have graded 0/1350 XSTest records each; harmful actionability
and benign helpfulness remain unvalidated. There is no
independent *external* compute-host reproduction, operational
credential-rotation approval, responsible-publication signoff,
or model release approval. This capability benchmark does not
prove zero refusal or safe compliance.
