# XSTest: independent third-human adjudication handoff

**Current state (2026-10-30):** both original reviewer assignments remain
at 0/1,350 graded responses. The third-reviewer queue MUST NOT be created
from these incomplete packets. This protocol is ready for a future
authorized, human-only independent review. No human labels are supplied
or inferred by software.

The original three checkpoints and 450-source-prompt/1,350-blinded-response
study remain unchanged. The custodian privately retains the model unblinding
key. The third reviewer receives only a restricted CSV of disputed
original prompt/response pairs, not the private model key, assigned model,
or either original human's judgments.

## Prerequisites and privacy

Two distinct humans independently complete the existing 15 batches x 90
responses, each using the offline local_reviewer_terminal with
a unique reviewer ID. The original private batches are never overwritten
by adjudication. The source-custody audit verifies all original text,
response IDs, full 450 x 3 mapping, and owner-only file permissions.
Any missing primary human labels or inconsistent reviewer identity
cause fail-closed refusal.

The adjudication queue includes any disagreement over response category,
harmful actionability, or benign-only helpfulness, plus any original
human answer marked unclear/uncertain. Software NEVER chooses the
winning label, including for simple disagreements. An independent
third human must explicitly grade every disputed response, following
the same rubric and displaying only the blinded prompt and response.
An unresolved third verdict prevents final output.

## Local-only custodian workflow

Run from /home/scott/git/lfm3-refusal-study-20261029 in the trusted
local shell with the existing Python environment:

    python -m src.reviewer_handoff_audit \
      --packet-dir results/xstest_full_review_private \
      --batches-dir results/xstest_review_batched_private

    python -m src.reviewer_adjudication prepare \
      --packet-dir results/xstest_full_review_private \
      --batches-dir results/xstest_review_batched_private \
      --adjudication-dir results/xstest_adjudication_private

The restricted adjudicator at a trusted interactive terminal then uses:

    python -m src.reviewer_adjudication review \
      --adjudication-dir results/xstest_adjudication_private \
      --adjudicator-id THIRD_HUMAN_LOCAL_ID

If a human must revisit an unresolved response, use --index N
(1-based index in the restricted adjudication queue). The writer
takes the same private directory lock and atomic save protections
as the existing review terminal.

Only after the third review has resolved all disputes, the custodian
can generate a new, PRIVATE full paired analysis:

    python -m src.reviewer_adjudication finalize \
      --packet-dir results/xstest_full_review_private \
      --batches-dir results/xstest_review_batched_private \
      --adjudication-dir results/xstest_adjudication_private \
      --output results/xstest_human_adjudicated_private.json

All output paths must be new; no evidence is overwritten. If the
original review files change after queue creation, finalization
fails on SHA-256 mismatch and requires a new explicit handoff.
The adjudicator must have a consistent identity different from
either of the two original reviewers.

The report contains only stratified aggregate paired effects,
not prompt texts, generated responses, individual reviewer
identities, private source file paths, or the model unblinding key.
Private source SHA-256 fingerprints remain in the owner-only
manifest for custodial verification.

**Research limitations:** This procedure does not validate that
a claimed human identifier corresponds to an independent person;
the custodian must verify that out of band. The existing standard
two-reviewer publication gate does not accept adjudicated evidence
as automatically approved. An independent scientific and safety
review is still required before public aggregate publication,
any zero-refusal claim, model release, or production use.

All tests of this procedure use fabricated reviewer labels ONLY
inside disposable synthetic fixtures. They do not count as human
review of the actual checkpoint outputs.
