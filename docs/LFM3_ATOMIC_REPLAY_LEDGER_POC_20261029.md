# LFM3 Release Evidence: Local SQLite Replay Ledger POC

This research layer stacks on draft PR #79's Ed25519 verifier and the
prior fail-closed registry/export changes. It is a **non-authorizing,
local-only** replay recorder. It does not enable model promotion, export,
training, package publication or any live service action.

## Implemented

src/lfm3_release_replay_ledger.py first calls the read-only Ed25519
verifier with a trusted-public-key mapping and expected manifest
supplied by a separate caller. Signatures must verify, timestamps and
manifest bindings must pass, and the receipt cannot include unsigned
model metadata or embedded trusted keys. The verifier's prior read-only
set of release IDs is deliberately empty in this wrapper: SQLite's
transactional primary key enforces the one-time local claim instead.

Successful verification alone never claims an ID. The caller must
supply an existing, private 0700 directory in an absolute path
(no symlink components). Ledger file is a local SQLite database,
release_claims.sqlite3, owned by the running user and chmod 0600.
Existing symlinks, hardlinks, directories, or incompatible schemas at
the database path are rejected.

SQLite uses BEGIN IMMEDIATE, synchronous=FULL, journal_mode=DELETE,
and a WITHOUT ROWID table whose primary key is the 32-hex-character
release ID. A competing writer of the same ID gets a blocked reason
release_id_already_claimed, and an interrupted transaction does not
consume the release ID. All exceptions are translated to non-sensitive
blocked statuses. The ledger stores only the release ID, a SHA-256
envelope digest, and the UTC claim time; not prompts, private paths,
keys, model weights or full receipts.

Result for a successfully claimed signed proposal:

    status: local_claim_recorded_non_authorizing
    signature_verified: true
    claim_committed: true
    local_replay_claimed: true
    fleet_wide_replay_protected: false
    artifact_custody_verified: false
    heldout_quality_attested: false
    export_permitted: false
    promotion_permitted: false

A valid cryptographic signature does not attest provenance, verify
the content behind a claimed hash, demonstrate independent holdout
quality or grant the signer any trusted release authority. The
existing exporter and registry still return HOLD.

## Tests / resource constraints

All tests operate in Pytest temp directories. Research test keys are
generated ephemerally in RAM; no signing key is written to disk or
registered as a production root. Positive/negative tests exercise:
concurrent threads, competing local processes, first-claim semantics,
persistent reopen, unique IDs, invalid signatures/requests, private
folder permissions, symlink/hardlink substitution, schema corruption,
uncommitted transaction crash recovery, no permission bypass, no
artifact writes and no output disclosure.

No production database, network, GPU, model weights or model files
are accessed. SQLite is part of Python's standard library and
requires no external service.

## Important limitations before a production release

- Atomicity and replay exclusion hold only for cooperating writers
  using this same SQLite file on a supported local filesystem.
  Separate nodes, snapshots, restored backups, copies or arbitrary
  database deletion permit replay. No fleet-wide ledger is implemented.
- SQLite synchronous=FULL and transactions do not guarantee physical
  storage survival across all crash/power-loss and filesystem
  configurations. Use reviewed storage with durable replicated
  consensus and explicit fault injection in production.
- The filesystem prechecks are NOT fd-anchored; privileged adversaries
  or a process controlling path components can race path replacement.
  Do not put this ledger in a shared/untrusted directory.
- The signing authority and trust root are not provisioned, and
  operator approval/independent heldout evaluation remain separate
  blockers. Passing this local claim cannot enable export.
- Cross-process ledger encryption, secrets protection, tamper-evident
  audit logs, key rotation, release rollback and an atomic staging /
  publish transaction are not implemented.
- Do not drop, rewrite or migrate the database in production without
  an independently reviewed custody / retention plan.
- The existing full-repository 69% coverage threshold is unchanged;
  this POC must pass exact-head hosted CI without threshold reductions.

The eventual system should provide a separately authenticated release
authority, a fleet-wide linearizable one-time approval ledger,
artifact descriptor and digest verification, independent evaluator
attestation, human approval and rollback-aware atomic publication.
