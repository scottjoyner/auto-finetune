# LFM3 private pilot: attested-route single-snapshot integrity

**October 31, 2026 — x1-370, draft PR #103**

The frozen model-blinded 32-source-prompt / 96-response-per-reviewer
pilot requires human review of its original mode-0600 private route
and 15×90 private batch files. Before this hardening, the status/review
entry point and post-review agreement reporter verified a private
route against the frozen 1,350-generation source, then reopened the
route path to obtain entries. Those are separate file reads; a route
replacement between them could cause use of a route other than the
one that passed the exact-source attestation.

## Verified fix

- `load_route(..., include_manifest=True)` now opens the private
  route once with `O_NOFOLLOW`, checks the open descriptor with
  `fstat` for regular-file, owner-only and single-hardlink constraints,
  parses the manifest, and returns its entries **from the same
  in-memory object**. The default legacy return remains entries-only.
- `_regular_private` rejects a symlink at any route path component,
  including a nested intermediate ancestor.
- `attest_route_with_entries` validates the original frozen journal,
  source diversity evidence, three checkpoint hashes, exact
  deterministic 32-prompt/96-target selection, and original batch
  roster against that **single read**. It returns an aggregate
  attestation and the in-process private entries to trusted callers.
- The public `attest_route` CLI still returns only the aggregate
  attestation. No route IDs, prompt/model text, source classes,
  reviewer identities or per-reviewer labels appear in its output.
- `private_review_pilot status/review` and
  `private_pilot_agreement` now consume precisely the attested
  snapshot instead of reopening the route file later.
- Reviewer batch source/identity checks, local interactive-only
  human review, explicit two distinct reviewers, and prohibition
  on automated label generation are unchanged.

## Real host checks

Using the *existing original private route*, the updated
`private_review_pilot status` passed source attestation for
**reviewer 1 (0/96 graded)** and **reviewer 2 (0/96 graded)**.
The new agreement CLI independently ran that attestation, correctly
returned exit code **2** for incomplete independent human ratings
and created **no private agreement report**. The original
frozen journal and human review CSVs were not edited.

The full regression suite passed **391/391 tests**. New synthetic-only
tests prove that both the reviewer status and agreement reporter
consume the attested route even if a subsequent route read would be
different or fail, as well as rejecting intermediate symlinks and
multiply linked route files. The live pilot was neither rated
nor unblinded during this work.

**Limitations:** The checked snapshot does not prevent another
process from editing the original batch CSVs after custody checks.
Other source files must still be protected with their own provenance
checks; this fix is scoped to one avoidable route reread. It does
not produce behavioral-safety evidence, approve a checkpoint,
authorize a weight release, or resolve the host's storage incident.

The host's root filesystem remained at 100% df utilization with
~5.3 GiB available to ordinary users, under the separate
20 GiB heavy model-write floor. No new inference, checkpoint
copy, cleanup, ext4 reserve change or production service cutover
was performed.
