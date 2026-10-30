# Complete locally pinned ARC-Challenge results — October 30, 2026

All 1,172 questions of the SHA256-pinned ARC-Challenge test split were
ACTUALLY scored for each of exactly three frozen LFM3-1.7B checkpoints
using EleutherAI lm-eval 0.4.13. Each arm used 4,687 CPU bfloat16
choice-likelihood requests, or 14,061 across the experiment.
The independent aggregate validator reconciled all 3,516 case-level
binary outcomes against the pinned dataset and the original
evaluator scores and standard errors.

**These are full LOCAL-ADAPTED task results**, not canonical unmodified
Hub leaderboard results. An isolated task adapter inherited the
official ARC multiple-choice prompt/answer choice scoring and separate
acc and acc_norm metrics, but used a SHA256-pinned local parquet
because the official Hugging Face dataset was not cached offline.

## Actual full-test scores

| Checkpoint | Accuracy / correct count | Normalized accuracy / count |
|---|---:|---:|
| A: original instruction-tuned | 23.55%, 276/1172 | 26.11%, 306/1172 |
| B: M1 rank-one edited | 22.61%, 265/1172 | 25.34%, 297/1172 |
| C: pretrained base | 39.16%, 459/1172 | 42.83%, 502/1172 |

Standard errors verified from all private binary outcomes:
A acc .01239945, acc_norm .01283552;
B acc .01222420, acc_norm .01271090;
C acc .01426412, acc_norm .01446050.

## Matched A/B comparison

| Metric | B newly correct | B no longer correct | B minus A | Exact paired p | Six-test Holm p |
|---|---:|---:|---:|---:|---:|
| acc | 18 | 29 | -0.94 percentage points | 0.143865 | 0.287730 |
| acc_norm | 14 | 23 | -0.77 percentage points | 0.187742 | 0.287730 |

Neither A/B difference reaches 0.05 after correcting over the six
exploratory comparisons (two metrics times three matched model pairs).
These tests **do not establish equivalence, noninferiority, or
capability preservation**. C scores approximately 15.6–17.5 percentage
points higher than A/B, but the pretrained base has a different
post-training status, confounding a causal inference about editing.

## Evidence custody and limitations

The public aggregate-only evidence is at
evidence/20261030_arc_local_full1172_cpu_aggregate.json.
It contains all three original SHA256 checkpoint identities, the
test parquet and local adapter hashes, the inherited official
parent task YAML hash, per-arm private source JSON/JSONL file SHA256
receipts, exact statistics, matched-count results and explicit
release-denial flags. No private prompts, response text, answer keys
or sample-level model outputs are exposed.

All original sample JSONL and results JSON remain confidential on
x1-370 under results/lm_eval_arc_local_private. Their privacy status
is unaffected by publication of the aggregate report. The model
weights, roster, services, optimizer, training, drivers and production
deployments have not been modified.

**Remaining gates:** the unmodified canonical official Hub task
was not run, external independent reproducibility is unconfirmed,
and both real blinded human reviewer assignments remain ungraded
(0/1350 each). This capability experiment does not establish
behavioral zero refusal, benign helpfulness, harmful actionability,
safe compliance, human review, or model-release approval.
