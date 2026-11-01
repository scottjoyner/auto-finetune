# LFM3-1.7B: Immutable human correction requests and adjudication (2026-11-01)

**Implemented as a human-only, private extension of draft PR #103. No human scores have been entered.** The original 450 XSTest source prompts × frozen A/B/C model completions remain unchanged and model identity remains blinded.

## What this fixes

The interactive stage-one terminal now **defers helpfulness scoring entirely**. It records only refusal, actionability and optional uncertainty notes. Benchmark-benign helpfulness is gathered from the real human only after all 1350 first-stage ratings are entered, using the existing gated `src.reviewer_helpfulness_pass` tool. The adjudicator's separate third-human interface retains the combined rating prompts because their queue may contain cases requiring final benign helpfulness.

Once a human enters a first-stage rating or a second-stage helpfulness rating, ordinary review commands do not overwrite it. When the original reviewer identifies a possible mistake, the new `src.human_review_corrections` tool creates an **immutable request** in the packet's separate owner-only directory. Each request contains only blinded review ID, reviewer slot, original evidence row SHA-256, source batch/index, fixed reason category and false unblinding/release flags; it does not copy prompt or response text, model arm, reviewer identity, new grades or private notes. Requests use mode 0600 and exclusive creation; duplicates fail.

## Private operator procedure

Open a real authorized **local TTY**, not an agent session or logs. The reviewer must supply their own existing ID. Only after they have actually reviewed an item and discovered a possible mistake, run:

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Example arguments only; substitute the person's actual reviewed batch/row.
"$PY" -m src.human_review_corrections \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --reviewer 1 --batch 1 --index 2 \
  --reviewer-id '<the-original-human-id>' --reason refusal
```

Reason is one of `refusal`, `actionability`, `helpfulness`, `uncertainty`. A request for an ungraded row, wrong human ID, invalid range, changed source evidence, unauthorized file mode or duplicate existing request is rejected. A request does not change the actual rating or decide its replacement.

**The next independent third human**, once both original reviewer assignments are completed, must adjudicate all ordinary rating disagreements **plus every uniquely flagged correction request**, even if the two originals agree. `src.reviewer_adjudication queue` freezes request IDs and SHA-256 fingerprints alongside the original reviewer-source files. `finalize` re-verifies the original human entries and the correction request ledger; any added, missing, changed or symlinked request causes a hard failure before publishing the private report. The report distinguishes true reviewer disagreements from correction-requested cases. Per-case reviewed grades and model mapping remain private.

## Tests and release gates

Synthetic tests exercise append-only request creation, 0600 owner-only custody, source-row integrity, duplicate rejection, role identity mismatch, invalid names/paths, concurrent or subsequent mutation detection, and actual third-human adjudication of a **consensus-but-flagged** rating. No real reviewer answers were generated to make the tests pass.

This is a proposal-and-independent-adjudication mechanism, not a mechanism for silently overriding a human rating. It provides file provenance and distinct reviewer-ID checks; it does **not prove that two IDs represent two different real people** or establish OS-level segregation between users sharing a Unix account. Reviewer independence must also be established operationally. No model checkpoint may be changed or released from an activation score or ungraded source.

Latest real reviewer custody count at the start of this increment: 0/1350 human grades for each of two independent reviewer assignments. PR #103 stays draft and storage incident #106 remains open.
