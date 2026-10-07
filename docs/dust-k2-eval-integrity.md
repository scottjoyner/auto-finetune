# K2-Horizon evaluation-integrity gate

This research slice follows the cached-tail efficiency work. It does not change
the trainer or model. Its purpose is to ensure that a "heldout row count" is
not accidentally treated as an independent-task count when many rows share the
same user prompt.

## Problem found in the previous heldout

The previous K=1024 quality experiments used rows from existing heldout JSONL
files after excluding normalized prompts that also appeared in
`train.combined`. That train/eval separation is necessary but not sufficient:
different assistant responses to the **same heldout user prompt** can inflate
the apparent size of the evaluation set.

A read-only Xwing audit of tokenizable (<=512 token) prompt-disjoint pairs
found:

| response eval | prompt-disjoint pairs | unique normalized prompts | repeated rows |
| --- | ---: | ---: | ---: |
| `held-out-hermes-reasoning.jsonl` | 17 | **2** | 15 |
| `held-out-combined.jsonl` | 10 | **1** | 9 |

For the Hermes-reasoning file, one prompt accounts for 16 of the 17 usable
pairs. Therefore the earlier "12 heldout rows" must **not** be interpreted as
12 independent questions/tasks.

This does not invalidate the compute/fidelity measurements in PR #11. It does
mean the heldout-quality signal is much weaker than its row count suggested.

## Fail-closed selector change

`select_disjoint()` now requires the chosen evaluation examples to have
**unique SHA-256 hashes of normalized user prompts**, in addition to excluding
all prompts present in the training source.

The selector schema changes from pair-ranked v1 to
`sha256-ranked-unique-normalized-prompt.v2`. Reports include only hashes,
counts and aggregate token totals; no prompt/response text is serialized.

Live acceptance against `held-out-hermes-reasoning`:

- request 2 eval examples: **admitted**, 2 unique prompts, 131 scored assistant
  tokens, 15 repeated-prompt pairs excluded;
- request 3: **blocked** — not enough distinct tokenizable prompts;
- request 12: **blocked** for the same reason.

That failure is intentional. Future experiments cannot silently recover the
old inflated heldout size by choosing multiple responses to one prompt.

## Existing independent evaluation assets are much healthier

The repo already contains curated probes and concrete agentic benchmark tasks.
The same audit compares their normalized prompt hashes against every prompt in
`train.combined`, without emitting raw text.

| source | rows | unique prompts | exact prompt overlap with training |
| --- | ---: | ---: | ---: |
| `eval/probe.jsonl` | 15 | **15** | **0** |
| `eval/tasks/auto-verified.jsonl` | 49 | 32 | 1 |
| `eval/tasks/minicpm5.jsonl` | 3 | **3** | **0** |
| `eval/tasks/sample.jsonl` | 23 | **23** | **0** |

Across these four sources there are 90 source rows. After collapsing
within-file duplicate prompts, there are 73 prompt entries. Exactly one prompt
entry overlaps the training corpus. Removing it leaves **72 distinct,
cross-source unique normalized prompts**, with no remaining exact prompt
duplicates across the candidate union.

This is a substantially better basis for task-quality evaluation than the
response-heldout files.

## Scope of the audit

The audit proves only **exact normalized-prompt** separation and prompt
diversity. It does not prove semantic independence. Two differently worded
prompts can still describe the same task. Before treating the 72-prompt union
as a publishable benchmark, semantic/near-duplicate review should be added.

The agentic task files also have a different evaluation contract from
heldout CE: they specify prompts plus concrete sandbox checks, not gold
assistant responses. They are appropriate for behavior/task-completion
measurement rather than token-loss scoring.

## Evidence

The latest no-text audit is stored on Xwing local SSD:

`/media/scott/data/finetune-staging/eval-reports/dust-k2-eval-integrity-unique-v3-20261007.json`

SHA-256:

`e6efab9c9f291309927c3c9d9a4838b3e4dbe242bbd0bff668ae1cfc23a0c2d0`

The file is mode 600. The audit contains hashes and counts, not prompt,
response, or token text.

## Next gate

Do not spend the recovered PR #11 compute budget simply extending the old
response-heldout experiment.

Instead:

1. adapt the existing probe/task evaluation to K2-Horizon's native tool-call
   format (`<ifm|tool_calls>` / `<ifm|tool_call>`);
2. exclude the one exact training-overlap task and preserve a fixed task
   manifest;
3. add semantic/near-duplicate checks across that fixed candidate suite;
4. evaluate base versus in-memory forward-only LoRA on the same prompts and
   sandbox checks;
5. only then run predeclared 8–32 update schedules.

Checkpoint promotion remains blocked until independent task-quality improves
repeatably.
