# LFM3 review-pilot integrity and storage guard — October 30, 2026

**Measured on x1-370; both mechanisms are read-only until an actual
authorized human explicitly enters ratings.** No LFM3 checkpoints, human
grades, original source prompts, full-XSTest generations, private reviewer
sheets, or other project files were modified by this audit.

## Source-bound pilot attestation

The 32-prompt / 96-response-per-reviewer pilot already checked that every
blinded ID referred to an existing batch row. That was insufficient:
a legitimate but swapped ID, or a valid 96-record replacement sample,
could pass the old row/ID check and silently change pilot stratification.

`src/private_review_pilot_attestation.py` now reconstructs the original
selection from the frozen **1,350-generation** source journal, the pinned
450-prompt dataset, the source response-diversity evidence, and the private
model-unblinding map. It checks the precise eight-per-bucket selection
(safe/unsafe × C modal/other), three frozen arms for each source prompt,
96 distinct blinded targets, exact SHA-256-seeded order and every original
15×90 reviewer-batch location. It verifies model and source hashes,
private route source hashes, and original batch custody.

The established `src.private_review_pilot status` and `review` commands
now call this full validation BEFORE showing progress or entering any
ratings. A deliberate live negative test on a disposable route copy
proved the prior row-only checker would accept reordered valid IDs,
while the new source-bound check rejected them. The real mode-0600
route and all original human grading CSVs were left unchanged.
## Re-run on the authorized x1-370 host

From `research/lfm3-refusal-validation` in the current research branch:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
BASE=/home/scott/git/lfm3-refusal-study-20261029/results
ROUTE="$BASE/lfm3_private_review_pilot_20261030/route.json"

$PY -m src.private_review_pilot_attestation \
  --results-dir "$BASE/xstest_full_20261029_v1" \
  --packet-dir "$BASE/xstest_full_review_private" \
  --batches-dir "$BASE/xstest_review_batched_private" \
  --route "$ROUTE"

$PY -m src.private_review_pilot status \
  --packet-dir "$BASE/xstest_full_review_private" \
  --batches-dir "$BASE/xstest_review_batched_private" \
  --route "$ROUTE" --reviewer 1
```

Human review remains a **local TTY-only workflow** requiring real,
independent reviewers and their own IDs. No model judge, lexical-rule
substitution, copied grades, automatic unblinding, or publication of raw
prompt/response content is permitted. A completed pilot is not a complete
1,350-case independent review and cannot certify model refusal safety.
## Conservative local disk-capacity preflight

The new optional `src.research_storage_preflight` reads `statvfs`
available bytes and checks a planned write plus a protected reserve
against a minimum free-space floor. With the default example settings
(5 GiB anticipated writes, 8 GiB reserve, 20 GiB floor), the real host
measured **8.641 GiB free** and correctly returned exit code **2**
(`large_write_allowed=false`). The separate mounted /nas path had
approximately 36,252.873 GiB reported available. No files were moved.

```bash
$PY -m src.research_storage_preflight \
  --target /home/scott/git --planned-write-gib 5 \
  --reserve-gib 8 --minimum-floor-gib 20 --nas /nas
```

For large writes the operator must explicitly chain the preflight
as a prerequisite, e.g. `python -m src.research_storage_preflight ... &&
YOUR_AUTHORIZED_LFM3_JOB`. This is **not a global enforcement mechanism**:
external processes and unrelated agents can still consume root space.
Avoid copying weights or launching lengthy training until the free space
threshold and any needed staging capacity are verified.

Filesystem reporting does not reserve disk. Concurrent processes may
consume space after the check (TOCTOU). The NAS being mounted and having
space does not establish migration correctness; verify checksum, access
rights, filesystem behavior and service dependencies before moving data.
Do not delete files or alter other worktrees to satisfy this guard.
## Evidence and gates

Public `evidence/20261030_review_route_storage_preflight.json` contains
only aggregate route fidelity and sampled filesystem capacity. It
contains no blinded source IDs, private reviewer identities, source
prompt/response text, or model weights.

The real source-bound attestation passed for 32 prompts / 96 targets
and both original reviewer assignments. The pilot remains 0/96 graded
per reviewer and the full 1,350-row independent reviews are incomplete.
Tests of tampered selection, changed ordering, missing checkpoint arm,
missing manifest rows, capacity thresholds and invalid storage inputs
fail closed. The safety, scientific, publication and weight-release
gates remain blocked pending independent behavioral evaluation.
