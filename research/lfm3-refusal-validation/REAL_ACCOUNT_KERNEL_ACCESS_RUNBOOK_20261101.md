# LFM3: Actual reviewer OS access-denial runbook

**Engineering contract and synthetic-file tests only, November 1–2, 2026.**
No real separate human review accounts have been provisioned or tested.
The tool creates no accounts, changes no permissions, reads no source data,
and does not attest independent people or approve model release.

## What the tool proves and cannot prove

The module src.reviewer_kernel_access_probe provides a read-only two-step
workflow for an authorized custodian and independently operated reviewers.

First, preflight checks the canonical model-key path and EXISTING private
reviewer-owned source folders. It rejects missing targets, symlinks anywhere
in the path, hardlinks, nonregular samples, group/other access, common Unix
UIDs, key-owner reuse and root ownership. A two-reviewer preflight can run
before a third adjudicator exists. Once a third genuinely exists, rerun with
that third private source directory to check three distinct non-root UIDs.
A preflight only validates source identity and owner metadata, NOT denied
kernel access.

Second, run probe under each designated real reviewer login, on the SAME
host and mount namespace. It validates Linux real/effective/saved/fsuid
and no effective capabilities. It opens and immediately closes the
reviewer's own private batch as a positive control, then tries to open the
other original reviewer's file, the model key, and the third reviewer file
if present. Only actual EACCES/EPERM counts as denial; missing paths,
unexpected I/O failures and a readable restricted target block success.
The probe NEVER calls read() on any sensitive file. It prints sanitized
boolean/count results, not model keys, prompts, source IDs or reviewer IDs.

A successful JSON response is NOT signed audit evidence. Administrator
observation of the exact process UID and comparison to the trusted preflight
paths is REQUIRED before anyone claims the OS test was performed correctly.
Do not automatically transform any positive test into human identity
attestation, model safety evidence or release authorization.

## Required administrator command sequence

From the research directory on x1-370, with the approved Python venv,
and ONLY after genuine owner-only reviewer workspaces have been provisioned:

    PY=/home/scott/lfm3-research/train-venv/bin/python
    PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private

    "$PY" -m src.reviewer_kernel_access_probe preflight \
       --packet "$PACKET" \
       --reviewer-one '<real-reviewer-one-private-workspace>' \
       --reviewer-two '<real-reviewer-two-private-workspace>'

To include a third genuinely independent reviewer, supply:

       --adjudicator '<real-third-reviewer-private-initial-workspace>'

The trusted administrator must separately run probe as each actual non-root
reviewer, NEVER as root or the model-key custodian:

    "$PY" -m src.reviewer_kernel_access_probe probe \
       --role 1 \
       --expected-uid '<actual-role-one-numeric-uid>' \
       --own-sample '<real-reviewer-one-private-workspace>/batch_001.csv' \
       --other-first '<real-reviewer-two-private-workspace>/batch_001.csv' \
       --key "$PACKET/private_model_key.json"

For role two, swap the first/second reviewer paths and pass role 2 and
their actual UID. After third-person provisioning, roles one and two also
pass --other-second pointing at the real third reviewer's
blinded_adjudication.csv. The role-three invocation MUST specify both
original reviewer sample files as --other-first and --other-second and its
own separately owned blinded_adjudication.csv as --own-sample.

The current original shared-UID reviewer batches MUST NOT be used as
stand-ins for independently transferred human-owned workspaces. The
administrator must separately verify true personhood, trusted account-to-
person assignment, actual cross-account access-denial logs, and access
from other hosts/paths/services. Do not fabricate a third account merely
to make the test turn green.

## Evidence and limitations

The successful two-person preflight explicitly reports third reviewer
PRESENT=false and THREE distinct owners=false. The one-other-reviewer
probe reports one tested path, never that both original reviewers have
been blocked. Role three cannot claim success until both original review
targets and the key are denied.

The unit tests exercise real kernel open denials on harmless temporary files
as well as mocked UID ownership to test contracts; they are NOT live three-
person OS proof. No private model-key contents, other review contents,
new human scores or model weights are ever accessed or changed.

Remaining: actual authorized account setup and transfer, three independent
person attestations when an adjudicator is needed, true live OS denials
with trusted execution, human review and scientific approval.

Keep P0 issue #107 and storage issue #106 OPEN. PR #103 DRAFT.
No model release authorized.
