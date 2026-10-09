"""Restricted, receiver-service-UID-only SSH forced-command entry point.

INSTALLATION CONTRACT (not executed by this research tool):
  - Independent non-admin Unix service account on x1, different from 'scott'
    and from Xwing's SSH-accessible identity.
  - Root-owned, group/world NON-writable installed executable and package tree,
    e.g. /usr/local/libexec/dust-receipt-service.py.
  - Service-owned mode-700 data root at /var/lib/dust-receipt, a mode-600
    receiver-owned.key, inaccessible to producer SSH identity.
  - Service's authorized_keys entry uses restrict + command=<fixed executable>,
    no PTY, forwarding or shell. Restrict filesystem and process privileges.
  - NO fallback to scott's old receiver key. Prior receipts stay legacy.

Only accepts "receive <run-id-hex32> <prebatch-sha256-hex64>" via
SSH_ORIGINAL_COMMAND. The producer controls only these two opaque digests.
Never accepts arbitrary paths, setup, verify, shell, semicolons or flags.
This script MUST NOT be invoked from a scott-writable Git worktree in a
deployed receiver service; root-owned installed copy is mandatory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys

from .k2_receipt_receiver import canonical, receive

SERVICE_ROOT = Path("/var/lib/dust-receipt")
INSTALLED_PATH = Path("/usr/local/libexec/dust-receipt-service.py")
_ALLOWED = re.compile(r"receive ([0-9a-f]{32}) ([0-9a-f]{64})\Z", re.ASCII)


def parse_forced_command(command: str | None) -> tuple[str, str]:
    if not isinstance(command, str) or len(command) > 105:
        raise ValueError("missing/oversized SSH_ORIGINAL_COMMAND")
    match = _ALLOWED.fullmatch(command)
    if match is None:
        raise ValueError("refuse any non-receipt command")
    return match.group(1), match.group(2)


def verify_separate_identity(*, current_uid: int, service_uid: int,
                             producer_uid: int, service_key_uid: int,
                             data_root_uid: int, service_root_mode: int,
                             signing_key_mode: int) -> None:
    """Pure, testable policy. An explicit producer-read-denied probe is still
    required: a valid mode/UID tuple alone cannot prove remote SSH isolation.
    """
    ids = (current_uid, service_uid, producer_uid, service_key_uid, data_root_uid)
    if any(type(v) is not int or v < 0 for v in ids):
        raise PermissionError("invalid custody Unix principal")
    if service_uid == 0 or service_uid == producer_uid:
        raise PermissionError("service may not be root or producer identity")
    if current_uid != service_uid or service_key_uid != service_uid:
        raise PermissionError("receiver cannot sign outside service UID")
    if data_root_uid != service_uid:
        raise PermissionError("private receiver directory must belong to service")
    if (service_root_mode & 0o777) != 0o700:
        raise PermissionError("receiver private directory must be mode 700")
    if (signing_key_mode & 0o777) != 0o600:
        raise PermissionError("receiver signing key must be mode 600")


def verify_installed_code(path: Path, *, expected_path=INSTALLED_PATH) -> None:
    if path != expected_path or path.is_symlink():
        raise PermissionError("service must execute approved installed path")
    path = path.resolve(strict=True)
    st = path.stat()
    if (not stat.S_ISREG(st.st_mode) or st.st_uid != 0
            or st.st_mode & 0o022):
        raise PermissionError("service executable must be root owned and not writable")
    # Every ancestor after the root of filesystem must be immutable to
    # service and producer identities. An editable package import path
    # would let scott exfiltrate the service signing key.
    for parent in path.parents:
        ps = parent.stat()
        if ps.st_uid != 0 or ps.st_mode & 0o022:
            raise PermissionError("service executable ancestor writable")
    for module in (receive.__code__.co_filename, canonical.__code__.co_filename):
        module_path = Path(module).resolve(strict=True)
        module_stat = module_path.stat()
        if module_stat.st_uid != 0 or module_stat.st_mode & 0o022:
            raise PermissionError("imported receipt implementation writable")
        for parent in module_path.parents:
            ps = parent.stat()
            if ps.st_uid != 0 or ps.st_mode & 0o022:
                raise PermissionError("import package ancestor writable")


def run_service(*, original_command: str | None,
                installed_executable: Path,
                service_root: Path = SERVICE_ROOT,
                service_uid: int,
                producer_uid: int) -> dict:
    run_id, digest = parse_forced_command(original_command)
    verify_installed_code(installed_executable)
    if service_root.resolve() != SERVICE_ROOT:
        raise PermissionError("service cannot use caller-provided custody directory")
    st = service_root.stat()
    key_st = (service_root / "receiver-owned.key").stat()
    verify_separate_identity(
        current_uid=os.geteuid(), service_uid=service_uid,
        producer_uid=producer_uid, service_key_uid=key_st.st_uid,
        data_root_uid=st.st_uid, service_root_mode=stat.S_IMODE(st.st_mode),
        signing_key_mode=stat.S_IMODE(key_st.st_mode))
    return receive(service_root, run_id, digest)


def main():
    # service_uid and producer_uid are INSTALL-TIME pinned constants, not
    # request arguments or environment variables. Package must be patched
    # by an operator after provisioning and preserved in root-owned copy.
    service_uid = -1
    producer_uid = -1
    if service_uid < 0 or producer_uid < 0:
        raise SystemExit("RECEIVER_NOT_PROVISIONED: separate UID gate HOLD")
    receipt = run_service(
        original_command=os.environ.get("SSH_ORIGINAL_COMMAND"),
        installed_executable=INSTALLED_PATH,
        service_uid=service_uid, producer_uid=producer_uid)
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
