# LFM3-1.7B — key-free third-human benign-helpfulness handoff

**Research contract only; tested with synthetic human ratings.** No real third-human reviewer account has been provisioned, no real annotation has been entered or exported, and a separate Unix account/ID is not proof of a separate human. PR #103 stays draft; P0 #107 stays open.

## Review sequence and privacy

`src.verified_third_helpfulness` handles the third adjudicator's second review stage **only after all initial disputed-case refusal and harmful-actionability judgments are complete and resolved**. Any `unclear` or `uncertain` third-human first-stage judgment blocks phase two; it must be reconciled through a separately auditable process, not silently counted as resolution.

The custodian revalidates the complete original two-stage merged reviewer source, pinned 450-case benchmark, original account ownership of both first-stage reviewers, the provenance-bound third-adjudication escrow, and every third-human first-stage row. Only then can the custodian construct a NEW owner-only restricted staging workspace with exactly two files: `third_benign_helpfulness.csv` and `private_third_benign_manifest.json`. The CSV contains **only the subset of disputed blinded cases belonging to the 250 benchmark-safe prompts across three checkpoints**, which is between 0 and 750 rows depending on actual disputes. It contains the third adjudicator's already-entered initial judgments, their original blinded prompt/response text, and a BLANK helpfulness 0–3 field. It excludes the source model key, benchmark unsafe cases, original first/second reviewer grades and identities, and model checkpoint labels.

**The custodian-owned output is staging only.** An administrator must securely copy its two files into an already authorized, isolated private directory actually owned by the SAME independently operated third-human Unix account that completed the first-stage adjudication. Only that account, using a local interactive TTY, may enter helpfulness 0–3 scores. The system refuses root, the exporter and the two original reviewers as the third reviewer, and refuses a different UID from the third-human initial-review workspace. The source stage-one manifest and CSV byte digests are pinned; any change in the original adjudicator judgments, notes, filenames or file ownership invalidates the stage-two handoff.

Each completed helpfulness score uses the existing human-only, append-only save routine: private file permissions, a directory-level save lock, concurrency recheck and no overwrite of an already saved score. The custodian's **read-only** `verify-return` operation recomputes the exact benign disputed-case subset using the original pinned private key; it rejects unsafe additions, missing or duplicated IDs, forged private manifest source hashes, changed initial judgment context, changed original canonical files, incomplete human grades and wrong reviewer ownership. It does NOT import into canonical source, finalize third-human adjudication, compute human behavioral safety estimates or authorize a checkpoint release.

## Operator commands — illustrative only, NOT ready for live use

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
CANONICAL=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Approved custodian after original 2-stage review and third-human initial grades:
"$PY" -m src.verified_third_helpfulness prepare \
  --packet "$PACKET" --canonical "$CANONICAL" \
  --merge-dir '<verified-private-two-stage-merge>' \
  --escrow '<verified-private-third-human-escrow>' \
  --first-a '<completed-original-reviewer-one-workspace>' \
  --first-b '<completed-original-reviewer-two-workspace>' \
  --phase-one '<completed-third-human-initial-workspace>' \
  --destination '<approved-owner-only-custodian-parent>/third-benign-staging'

# Administrator transfers only the two-file benign staging bundle privately
# to the SAME independently operated non-root third reviewer account.

# Third HUMAN under their own Unix account and local interactive TTY:
"$PY" -m src.verified_third_helpfulness status \
  --workspace '<third-human-owned-benign-workspace>' \
  --phase-one '<completed-third-human-initial-workspace>'
"$PY" -m src.verified_third_helpfulness review \
  --workspace '<third-human-owned-benign-workspace>' \
  --phase-one '<completed-third-human-initial-workspace>' \
  --reviewer-id '<original-third-human-private-reviewer-id>'

# Approved custodian re-verifies ALL original and merged source custody:
"$PY" -m src.verified_third_helpfulness verify-return \
  --packet "$PACKET" --canonical "$CANONICAL" \
  --merge-dir '<verified-private-two-stage-merge>' \
  --escrow '<verified-private-third-human-escrow>' \
  --first-a '<completed-original-reviewer-one-workspace>' \
  --first-b '<completed-original-reviewer-two-workspace>' \
  --phase-one '<completed-third-human-initial-workspace>' \
  --workspace '<completed-third-human-owned-benign-workspace>'
```

## Remaining gates

Real independent personhood attestation for the two original human graders and third adjudicator; actual cross-account key/reviewer access denial; completed original 1,350×2 refusal/actionability, 750×2 benign helpfulness, real disputed-case adjudication and safe-subset human helpfulness; audited correction transfer if any; provenance-aware final merged consensus/statistics. If there are no actual disputes, an alternate explicitly verified no-adjudication path is needed; do not manufacture a third person or imaginary scores. These are not demonstrated by synthetic data.

Tests `tests/test_verified_third_helpfulness.py` verify key-free limited selection, owner transfer as a synthetic simulated UID, full initial human stage gate, original stage-one immutability, append-only help scores, forged ID/manifest rejection, canonical drift, insecure or hard-linked file rejection, unresolved `unclear`/`uncertain` gate and nonauthorization. Synthetic tests never prove that real independent humans exist.

**No release:** original frozen A/B/C model weights unchanged; actual reviewer packet files stay private and ungraded. Storage issue #106 and isolation P0 #107 remain open; draft PR #103 is not approved for production release.
