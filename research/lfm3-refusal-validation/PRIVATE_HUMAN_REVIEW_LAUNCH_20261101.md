# LFM3-1.7B: Private two-stage human reviewer launch (2026-11-01)

**Workflow prepared, human annotation NOT STARTED.** Both real reviewer assignments currently have 0/1350 independently graded cases. Original packet/key hashes, 30 original reviewer batches, and private permissions pass custody checks. Model releases remain blocked.

## Purpose and phase boundaries

First stage: two distinct real humans independently review each of the 1350 shuffled A/B/C responses, entering refusal category and harmful actionability without access to model identity or benchmark safe/unsafe labels. Benign helpfulness can be left empty in this stage. Human entries are stored atomically in private 0600 CSVs; a previously submitted rating cannot silently be overwritten.

Second stage: only **after that individual reviewer has entered all 1350 initial refusal/actionability ratings**, the reviewer_helpfulness_pass tool enables human 0–3 usefulness ratings for the 750 benchmark-safe responses. It only discloses benchmark-benign eligibility to the approved reviewer after first-stage completion. It never exposes model identity, generates labels, modifies refusal ratings, or overrides prior helpfulness scores. The current real packets must FAIL this admission check.

## Execute only on a local authorized interactive terminal on x1-370

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# No prompts, model responses or unblinding keys printed by custody audit.
"$PY" -m src.reviewer_handoff_audit --packet-dir "$PACKET" --batches-dir "$BATCHES"
"$PY" -m src.local_reviewer_terminal --batches-dir "$BATCHES" --reviewer 1 --list-batches
"$PY" -m src.local_reviewer_terminal --batches-dir "$BATCHES" --reviewer 2 --list-batches

# A real reviewer in their own local TTY, with a private reviewer ID:
"$PY" -m src.local_reviewer_terminal --batches-dir "$BATCHES" --reviewer 1 --reviewer-id '<reviewer-one-id>' --next-batch

# Once that same reviewer has completed all 1350 first-stage ratings:
"$PY" -m src.reviewer_helpfulness_pass --packet-dir "$PACKET" --batches-dir "$BATCHES" --reviewer 1 --reviewer-id '<reviewer-one-id>' --status
"$PY" -m src.reviewer_helpfulness_pass --packet-dir "$PACKET" --batches-dir "$BATCHES" --reviewer 1 --reviewer-id '<reviewer-one-id>' --next-batch
```

Reviewer 2 separately uses --reviewer 2 and an independently chosen, **different** reviewer identity. Do not run human-review TTY operations through an agent, ChatGPT, network logs, or a pipe. Never copy raw prompt text, model responses, reviewer names, or private source/key JSON to GitHub.

After both original humans have completed the **entire 1350×2** assignment, use the existing third-human adjudication workflow for all disputes. The adjudicated private report now contains the human-only 18-pair matched activation/behavior bridge and the existing primary familywise 450-prompt comparison. Neither analysis is automatically populated from the current blank reviewer sheets.

## Evidence preservation and limitations

The second-stage helper rejects incomplete first-stage scores, mismatched reviewer identity, stale concurrent changes, manipulated source/key hashes, non-benign helpfulness edits, and attempts to overwrite an existing rating. The first-stage terminal now also prohibits silent replacement of a completed rating. If a human mistakes an entry, preserve the original as evidence and use a separately audited correction procedure; never silently edit an existing CSV rating.

Current evidence is unchanged: 0 real human labels; synthetic tests only for the reviewer code; checkpoints A/B/C frozen. PR #103 remains draft and storage issue #106 remains open.
