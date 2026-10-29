"""Offline, local SQLite replay claims for *verified, non-authorizing* evidence.

IMPORTANT: A durable local replay claim is NOT a release approval, a shared
fleet-wide replay barrier, independent model provenance, or export permission.
No production trust root, export integration or signer is provided.

A private caller-owned ledger directory must already exist. The module never
creates directories, imports model runtimes, opens model artifacts or logs
receipt contents.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Mapping

from src.lfm3_release_receipt_verifier import (
    verify_release_evidence,
)

SCHEMA = "auto-finetune.release-replay-ledger.v1"
DB_NAME = "release_claims.sqlite3"
CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS release_claims ("
    "release_id TEXT PRIMARY KEY NOT NULL,"
    "envelope_sha256 TEXT NOT NULL,"
    "claimed_at_utc TEXT NOT NULL"
    ") WITHOUT ROWID"
)


@dataclass(frozen=True)
class LedgerReceipt:
    status: str
    reason: str
    signature_verified: bool = False
    claim_committed: bool = False
    local_replay_claimed: bool = False
    fleet_wide_replay_protected: bool = False
    artifact_custody_verified: bool = False
    heldout_quality_attested: bool = False
    export_permitted: bool = False
    promotion_permitted: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        """Never return the release ID, signer, private paths or token."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class _LedgerUnavailable(Exception):
    pass


def _private_directory(folder: Path) -> None:
    """Reject obvious path substitution and unsafe sharing.

    This is a local research boundary, not a race-free fd-anchored security
    primitive. Privileged adversaries can still replace filesystem entries.
    """
    if not folder.is_absolute():
        raise _LedgerUnavailable("ledger_path_must_be_absolute")
    if not folder.exists() or folder.is_symlink():
        raise _LedgerUnavailable("private_ledger_directory_required")
    info = folder.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise _LedgerUnavailable("private_ledger_directory_required")
    # No symlink components in the caller-supplied path. The parent of a
    # private directory can be shared (e.g. /tmp with sticky bit).
    current = Path(folder.anchor)
    for component in folder.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise _LedgerUnavailable("ledger_path_symlink_not_allowed")


def _db_file_info(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()):
        raise _LedgerUnavailable("unsafe_ledger_file")


def _atomic_claim(
    folder: Path, release_id: str, envelope_digest: str, timestamp: str,
    *, busy_timeout_ms: int,
) -> str:
    """Exactly one cooperating SQLite writer wins the unique-ID insert."""
    _private_directory(folder)
    db = folder / DB_NAME
    _db_file_info(db)
    if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 5000:
        raise _LedgerUnavailable("invalid_busy_timeout")
    conn = None
    try:
        # Never create a database in an untrusted caller-selected directory.
        conn = sqlite3.connect(str(db), timeout=busy_timeout_ms / 1000,
                               isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA journal_mode=DELETE")
        _db_file_info(db)
        # The directory already limits access; protect DB bytes as well.
        os.chmod(db, 0o600, follow_symlinks=False)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(CREATE_SQL)
        # Detect schema substitution rather than silently writing to a
        # pre-existing view or incompatible table under the same name.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='release_claims'"
        ).fetchone()
        if row is None or row[0] != CREATE_SQL.replace("IF NOT EXISTS ", ""):
            conn.execute("ROLLBACK")
            raise _LedgerUnavailable("unexpected_ledger_schema")
        try:
            conn.execute(
                "INSERT INTO release_claims(release_id,envelope_sha256,claimed_at_utc) "
                "VALUES(?,?,?)",
                (release_id, envelope_digest, timestamp),
            )
        except sqlite3.IntegrityError:
            conn.execute("ROLLBACK")
            return "duplicate"
        conn.execute("COMMIT")
        return "claimed"
    except (OSError, sqlite3.Error) as exc:
        # No errors or caller-supplied paths returned in public results.
        raise _LedgerUnavailable("ledger_io_or_transaction_failed") from exc
    finally:
        if conn is not None:
            conn.close()


def claim_verified_release_evidence(
    envelope_bytes: bytes,
    *,
    trusted_public_keys: Mapping[str, Any],
    expected_request: dict[str, str],
    ledger_directory: str | Path,
    revoked_key_ids: frozenset[str] | set[str] = frozenset(),
    now_utc: datetime | None = None,
    busy_timeout_ms: int = 1500,
) -> LedgerReceipt:
    """Verify a signature and atomically record a non-authorizing replay ID.

    The caller MUST independently provision public keys and expected request.
    The verifier's read-only seen-ID snapshot is empty *here* because the
    SQLite transaction, not an external snapshot, enforces uniqueness.
    """
    check = verify_release_evidence(
        envelope_bytes,
        trusted_public_keys=trusted_public_keys,
        expected_request=expected_request,
        previously_seen_ids=frozenset(),
        revoked_key_ids=revoked_key_ids,
        now_utc=now_utc,
    )
    if not check.signature_verified:
        return LedgerReceipt(status="blocked", reason="evidence_verification_failed")
    try:
        if not isinstance(ledger_directory, (str, Path)):
            raise _LedgerUnavailable("private_ledger_directory_required")
        manifest = json.loads(envelope_bytes)["manifest"]
        release_id = manifest["release_id"]
        digest = hashlib.sha256(envelope_bytes).hexdigest()
        clock = now_utc if now_utc is not None else datetime.now(timezone.utc)
        if not isinstance(clock, datetime) or clock.tzinfo is None:
            raise _LedgerUnavailable("invalid_claim_clock")
        timestamp = clock.astimezone(timezone.utc).isoformat(timespec="seconds")
        state = _atomic_claim(
            Path(ledger_directory), release_id, digest, timestamp,
            busy_timeout_ms=busy_timeout_ms,
        )
    except (_LedgerUnavailable, OSError, sqlite3.Error, TypeError, KeyError, ValueError):
        return LedgerReceipt(
            status="blocked", reason="local_ledger_unavailable",
            signature_verified=True,
        )
    if state == "duplicate":
        return LedgerReceipt(
            status="blocked", reason="release_id_already_claimed",
            signature_verified=True,
        )
    return LedgerReceipt(
        status="local_claim_recorded_non_authorizing",
        reason="no_fleet_wide_replay_or_release_approval",
        signature_verified=True,
        claim_committed=True,
        local_replay_claimed=True,
        fleet_wide_replay_protected=False,
        artifact_custody_verified=False,
        heldout_quality_attested=False,
        export_permitted=False,
        promotion_permitted=False,
    )
