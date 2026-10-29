"""Read-only, descriptor-anchored artifact digest checks for signed research evidence.

Verify five independently supplied files against a signed proposal. This never
authorizes export, promotion, release, or model loading. No key, source code
or model blob is accepted from the signed envelope as a trust authority.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from src.lfm3_release_receipt_verifier import verify_release_evidence

DIGEST_FIELDS = frozenset({
    "base_model_sha256", "adapter_sha256", "training_code_sha256",
    "heldout_dataset_sha256", "evaluation_receipt_sha256",
})
BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
CHUNK_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024**3
MAX_TOTAL_BYTES = 16 * 1024**3


@dataclass(frozen=True)
class CustodyCheck:
    status: str
    reason: str
    signature_verified: bool = False
    five_content_hashes_verified: bool = False
    root_descriptor_anchored: bool = False
    artifact_source_independently_attested: bool = False
    heldout_independence_verified: bool = False
    fleet_wide_replay_protected: bool = False
    export_permitted: bool = False
    promotion_permitted: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class _Rejected(Exception):
    pass


def _root_fd(root: Path, stack: ExitStack) -> int:
    if not root.is_absolute():
        raise _Rejected("root_must_be_absolute")
    # The fleet's constrained process policy denies opening "/" directly.
    # Limit the absolute path's first component to reviewed local anchors,
    # opened with O_NOFOLLOW. This anchors all *subsequent* components to
    # descriptors, without silently accepting arbitrary filesystem roots.
    components = root.parts[1:]
    if not components or components[0] not in ("tmp", "home"):
        raise _Rejected("unreviewed_filesystem_root")
    fd = os.open("/" + components[0], os.O_RDONLY | os.O_DIRECTORY |
                 os.O_NOFOLLOW | os.O_CLOEXEC)
    stack.callback(os.close, fd)
    for component in components[1:]:
        if component in ("", ".", ".."):
            raise _Rejected("unsafe_root_component")
        fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY |
                     os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
        stack.callback(os.close, fd)
    st = os.fstat(fd)
    if (not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid()
            or st.st_mode & 0o077):
        raise _Rejected("private_owned_root_required")
    return fd


def _stat_identity(st: os.stat_result) -> tuple[int, ...]:
    return (
        st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_nlink,
        st.st_size, st.st_mtime_ns, st.st_ctime_ns,
    )


def _open_artifacts(root_fd: int, filenames: Mapping[str, str],
                    stack: ExitStack) -> dict[str, tuple[int, os.stat_result]]:
    if type(filenames) is not dict or filenames.keys() != DIGEST_FIELDS:
        raise _Rejected("invalid_filename_mapping")
    if any(type(name) is not str or not BASENAME.fullmatch(name) or
           name in (".", "..") or ".." in name for name in filenames.values()):
        raise _Rejected("unsafe_artifact_filename")
    if len(set(filenames.values())) != len(DIGEST_FIELDS):
        raise _Rejected("reused_artifact_filename")
    opened: dict[str, tuple[int, os.stat_result]] = {}
    identities: set[tuple[int, int]] = set()
    total = 0
    for field in sorted(DIGEST_FIELDS):
        # O_NONBLOCK prevents FIFO/device open from hanging before type rejection.
        fd = os.open(filenames[field], os.O_RDONLY | os.O_CLOEXEC |
                     os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        stack.callback(os.close, fd)
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or
                st.st_nlink != 1 or st.st_size < 0 or
                st.st_size > MAX_ARTIFACT_BYTES):
            raise _Rejected("unsafe_artifact_type_or_size")
        identity = (st.st_dev, st.st_ino)
        if identity in identities:
            raise _Rejected("reused_artifact_inode")
        identities.add(identity)
        total += st.st_size
        if total > MAX_TOTAL_BYTES:
            raise _Rejected("artifact_total_size_limit")
        opened[field] = (fd, st)
    return opened


def _digest_file(fd: int, before: os.stat_result) -> str:
    sha = hashlib.sha256()
    seen = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while seen < before.st_size:
        data = os.read(fd, min(CHUNK_BYTES, before.st_size - seen))
        if not data:
            raise _Rejected("artifact_truncated_during_read")
        seen += len(data)
        sha.update(data)
    if os.read(fd, 1):
        raise _Rejected("artifact_grew_during_read")
    if _stat_identity(os.fstat(fd)) != _stat_identity(before):
        raise _Rejected("artifact_changed_during_read")
    return sha.hexdigest()


def check_signed_artifact_bytes(
    envelope_bytes: bytes,
    *,
    trusted_public_keys: Mapping[str, Any],
    expected_request: dict[str, str],
    artifact_root: str | Path,
    filenames_by_digest_field: dict[str, str],
    revoked_key_ids: frozenset[str] | set[str] = frozenset(),
    now_utc=None,
) -> CustodyCheck:
    """Check manifest signature and actual local bytes; never grant authority.

    All five input files must already exist under a *trusted*, caller-owned
    0700 directory. Only simple names and anchored descriptors are used.
    The signed expected request is supplied independently, never inferred
    from untrusted contents. No output files or receipts are written.
    """
    verified = verify_release_evidence(
        envelope_bytes, trusted_public_keys=trusted_public_keys,
        expected_request=expected_request, previously_seen_ids=frozenset(),
        revoked_key_ids=revoked_key_ids, now_utc=now_utc,
    )
    if not verified.signature_verified:
        return CustodyCheck(status="blocked", reason="signature_or_request_invalid")
    try:
        if not isinstance(artifact_root, (str, Path)):
            raise _Rejected("invalid_root")
        with ExitStack() as stack:
            root_fd = _root_fd(Path(artifact_root), stack)
            inputs = _open_artifacts(root_fd, filenames_by_digest_field, stack)
            for field in sorted(DIGEST_FIELDS):
                fd, before = inputs[field]
                observed = _digest_file(fd, before)
                if observed != expected_request[field]:
                    raise _Rejected("artifact_digest_mismatch")
            # Catch alterations to earlier descriptors while hashing later
            # inputs. External writers remain a production-level threat.
            if any(_stat_identity(os.fstat(fd)) != _stat_identity(before)
                   for fd, before in inputs.values()):
                raise _Rejected("artifact_changed_between_hashes")
    except (_Rejected, OSError, ValueError, TypeError, OverflowError):
        return CustodyCheck(
            status="blocked", reason="artifact_custody_check_failed",
            signature_verified=True,
        )
    return CustodyCheck(
        status="local_hashes_match_signed_evidence_non_authorizing",
        reason="independent_evaluation_trust_and_release_approval_absent",
        signature_verified=True,
        five_content_hashes_verified=True,
        root_descriptor_anchored=True,
        artifact_source_independently_attested=False,
        heldout_independence_verified=False,
        fleet_wide_replay_protected=False,
        export_permitted=False,
        promotion_permitted=False,
    )
