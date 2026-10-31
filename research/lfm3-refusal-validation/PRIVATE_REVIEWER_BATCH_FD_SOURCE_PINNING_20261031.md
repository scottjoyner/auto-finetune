# Private XSTest reviewer batch: descriptor snapshots and pinned source

**2026-10-31, x1-370.** This follow-up hardens the existing
private, human-only blinded 450-prompt × three-arm review handoff.
It does not rate prompts, reveal arm assignments, modify the original
source journals, run models or approve a checkpoint.

## Threat and mitigation

Previously `open_batch` verified file mode, type and link count by
pathname, then opened the batch CSV or batch manifest again by pathname.
A same-path replacement between those actions could bypass the type or
privacy check. The stale-state check during interactive saving also
used a separate pathname-based CSV reader.

The new `_open_private_snapshot` in `src/local_reviewer_terminal.py`
opens the checked owner-only directory with a directory descriptor,
opens the child with `O_NOFOLLOW`, and verifies the open file using
`fstat`. It refuses symlink components, files accessible to group or
others, directories, and files with multiple hard links. The secure
JSON and CSV readers consume the already-open descriptor, not a
subsequent pathname lookup. CSV schema and uniqueness checks remain.

Within the model-blind pilot, `open_batch(..., packet_dir=...)` also
reads the original reviewer source CSV through a secure descriptor,
computes its SHA-256 from those exact bytes, compares that digest to
the original private 15×90 batch manifest, and requires every row's
`review_id`, `prompt` and `model_response` to match its position
in the original blinded source. Existing human-entered reviewer rating
fields are left untouched.

Pilot status, its interactive review path, and the post-review
agreement analyzer all opt into source-pinned batch loading. The
standalone batch-terminal mode can still be invoked without an
original packet path; these additional original-source comparisons
are therefore explicitly scoped to the pilot workflow. Normal,
cooperative concurrent human saves still use the directory lock,
fresh-content comparison and atomic replacement.

## Verification

- Live source-attested pilot status succeeded independently for both
  review assignments; **0 of 96 human ratings complete per reviewer**.
- Live agreement CLI returned exit code **2** with no report written,
  preserving the human-only readiness gate.
- Synthetic tests cover descriptor privacy, symlinks/hardlinks,
  non-owner-only modes, retention of the original verified file inode
  after an atomic pathname replacement, and no fallback to the
  pathname-based general CSV reader.
- New tests also verify that altered model response text with the
  **same valid blinded ID** is rejected against the original source,
  and that a modified source sheet fails its pinned SHA-256 check.
- Entire LFM3 study regression suite: **395/395 tests passed**.

## Remaining boundaries

These safeguards are not cross-process filesystem transactions and
cannot prevent a privileged adversary from changing independent
resources after the verified read. The broader `reviewer_handoff_audit`
has separate custody requirements; workflows not passing `packet_dir`
do not get source-column binding from this change alone.

The root filesystem remains severely constrained, with `/nas`
mounted separately. This work does not bypass the research write
headroom gate (20 GiB minimum), execute inference, change storage
reservations, or provide behavioral-safety outcomes. The pilot and
full-study human grading remain incomplete, and draft PR #103
remains unreleased.
