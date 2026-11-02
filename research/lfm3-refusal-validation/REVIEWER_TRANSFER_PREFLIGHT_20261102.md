# Reviewer transfer preflight — 2026-11-02

**Read-only implementation, synthetic tests only. No real account provisioning or human ratings.** PR #103 draft, P0 #107 open, storage #106 open.

## Readiness gate

The new reviewer_transfer_preflight module verifies the original frozen 450-prompt, three-arm study and blank canonical reviewer batches, then checks TWO existing owner-only staged exports, each exactly 15 batches of 90 prompts, all 1350 source records unchanged, with no reviewer names, scores, notes or private model key. Source ancestor symlinks, extra files, hardlinks and mismatched owners are rejected.

It requires two existing interactive non-root LOCAL Unix UIDs, different from each other and from the original model-key custodian UID. The local-account gate now requires each UID to have **exactly one matching entry in `/etc/passwd`**, and its username and shell must agree with the current NSS lookup; remote-only/LDAP-only identities, duplicate local UID entries, account-shell inconsistencies and noninteractive shells are rejected. This does not prove a real person's identity, consent, or actual login access. Their separately owned private destination parent directories must exist, be mode 0700 or tighter, and have enough free space for twice the staged bytes plus 1 MiB. Target paths must be absent, must not overlap originals, and must contain no symlink ancestors. It validates staging packets again after destination checks.

The gate is completely READ-ONLY: no mkdir, copy, chmod, chown, account changes, model weights, reviewer grades or original packets touched. Success returns account readiness only; transfer completed=false, real human identity and consent attested=false, kernel access denials verified=false, and release authorized=false.

## Operator command

Run only after authorized independent human reviewers have been provisioned and custodian-owned key-free stage-one exports exist:

    cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
    PY=/home/scott/lfm3-research/train-venv/bin/python
    export PYTHONDONTWRITEBYTECODE=1
    "$PY" -m src.reviewer_transfer_preflight \
      --packet /home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private \
      --canonical-batches /home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private \
      --first-stage '<existing-private-custodian-staging-one>' \
      --second-stage '<existing-private-custodian-staging-two>' \
      --first-destination '<first-account-owned-private-parent>/review' \
      --second-destination '<second-account-owned-private-parent>/review' \
      --first-uid '<authorized-first-account-numeric-uid>' \
      --second-uid '<authorized-second-account-numeric-uid>'

An approved administrator must separately perform and verify a secure owner-only transfer, then run actual reviewer-principal kernel access-denial tests against the private key and the other reviewer account under trusted supervision. A Unix UID and a positive preflight do not prove independent human personhood or consent. No account privileges or model keys should be handed to the reviewers.

## Tests and remaining human blockers

Thirteen synthetic tests cover valid packet readiness, UID reuse/key-owner reuse, account existence, sealed source tampering and labeling, swapped slots, destination overlap, symlink ancestors, existing paths, insufficient disk blocks, insecure parent, noninteractive account and altered canonical review. Synthetic UID mocks do not demonstrate real human isolation.

The live original reviewer samples and private key remain under one UID and each reviewer assignment remains 0/1350 rated. Real distinct human reviewers, their access attestations, actual reviews, third reviewer if disagreements require one, and scientific interpretation remain operational blockers. No checkpoint release.
