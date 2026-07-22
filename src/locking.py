"""Cross-process coordination for shared auto-finetune runtime resources.

Git worktrees isolate source edits.  These leases protect the shared staging tree,
GPU and checkpoint outputs that every worktree still uses.  Kernel ``flock`` is
the authority; JSON owner records are diagnostic only.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from src.config import Config


class ResourceBusy(RuntimeError):
    """Raised when another process owns a requested non-blocking lease."""


@dataclass(frozen=True)
class ResourceRequest:
    name: str
    shared: bool = False


class Lease:
    def __init__(self, lock_dir: str, request: ResourceRequest, *, command: str = ""):
        self.lock_dir = Path(lock_dir)
        self.request = request
        self.command = command
        self.lease_id = uuid.uuid4().hex
        self.fd: int | None = None
        self.owner_path: Path | None = None

    def acquire(self, *, blocking: bool = False) -> "Lease":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.lock_dir / f"{self.request.name}.lock"
        self.fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
        mode = fcntl.LOCK_SH if self.request.shared else fcntl.LOCK_EX
        if not blocking:
            mode |= fcntl.LOCK_NB
        try:
            fcntl.flock(self.fd, mode)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            owners = active_owner_records(str(self.lock_dir), self.request.name)
            detail = f"; owners={owners}" if owners else ""
            raise ResourceBusy(f"resource busy: {self.request.name}{detail}") from exc

        owners_dir = self.lock_dir / "owners" / self.request.name
        owners_dir.mkdir(parents=True, exist_ok=True)
        self.owner_path = owners_dir / f"{self.lease_id}.json"
        payload = {
            "lease_id": self.lease_id,
            "resource": self.request.name,
            "mode": "shared" if self.request.shared else "exclusive",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "process_start": _process_start(os.getpid()),
            "acquired_at": time.time(),
            "command": self.command or " ".join(sys.argv),
            "cwd": os.getcwd(),
        }
        _atomic_json(self.owner_path, payload)
        return self

    def release(self) -> None:
        if self.owner_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self.owner_path.unlink()
            self.owner_path = None
        if self.fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "Lease":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class LeaseSet:
    """Acquire several resources in deterministic order or release all."""

    def __init__(self, lock_dir: str, requests: Iterable[ResourceRequest], *, command: str = ""):
        merged: dict[str, ResourceRequest] = {}
        for req in requests:
            prior = merged.get(req.name)
            # Exclusive dominates shared when a command requests both.
            if prior is None or (prior.shared and not req.shared):
                merged[req.name] = req
        self.leases = [Lease(lock_dir, merged[name], command=command) for name in sorted(merged)]

    def acquire(self, *, blocking: bool = False) -> "LeaseSet":
        acquired: list[Lease] = []
        try:
            for lease in self.leases:
                lease.acquire(blocking=blocking)
                acquired.append(lease)
        except Exception:
            for lease in reversed(acquired):
                lease.release()
            raise
        return self

    def release(self) -> None:
        for lease in reversed(self.leases):
            lease.release()

    def __enter__(self) -> "LeaseSet":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def lock_dir(cfg: Config) -> str:
    return cfg.path("lock_dir")


def command_resources(cmd: str, label: str | None = None) -> list[ResourceRequest]:
    """Map a direct CLI command to the shared resources it mutates/consumes."""
    if cmd in {"extract", "hermes", "clean", "analyze", "mine-repairs", "dedup", "profile"}:
        return [ResourceRequest("harvest")]
    if cmd in {"format", "combine", "strata", "auto-balance", "eval-split",
               "pretokenize", "binarize", "dataset-version-restore"}:
        return [ResourceRequest("datasets")]
    if cmd == "all":
        return [ResourceRequest("harvest"), ResourceRequest("datasets")]
    if cmd in {"train", "dpo"}:
        checkpoint = f"checkpoint-{_safe_name(label or 'default')}"
        return [ResourceRequest("datasets", shared=True), ResourceRequest("gpu"),
                ResourceRequest(checkpoint)]
    if cmd in {"eval", "eval-all", "best", "sanity", "merge", "quantize",
               "bench", "bench-compare", "bench-matrix"}:
        return [ResourceRequest("datasets", shared=True), ResourceRequest("gpu")]
    if cmd in {"scheduler-run", "scheduler-loop"}:
        return [ResourceRequest("pipeline")]
    if cmd in {"deploy", "rollback"}:
        return [ResourceRequest("deploy")]
    return []


def command_leases(cfg: Config, cmd: str, label: str | None = None) -> LeaseSet:
    return LeaseSet(lock_dir(cfg), command_resources(cmd, label), command=f"src.cli {cmd}")


def legacy_training_processes() -> list[dict]:
    """Find trainers started before lease enforcement was installed.

    This is a migration guard, not the long-term lock authority. Once every
    trainer enters through this CLI, the kernel leases provide coordination.
    """
    found: list[dict] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        args = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        if "src.cli" in args and "train" in args:
            found.append({"pid": int(entry.name), "command": " ".join(args)})
    return found


def active_owner_records(lock_directory: str, resource: str | None = None) -> list[dict]:
    base = Path(lock_directory) / "owners"
    paths = list((base / resource).glob("*.json")) if resource else list(base.glob("*/*.json"))
    records: list[dict] = []
    for path in sorted(paths):
        try:
            rec = json.loads(path.read_text())
            rec["owner_file"] = str(path)
            records.append(rec)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def atomic_write_json(path: str | Path, payload: object) -> None:
    _atomic_json(Path(path), payload)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def _process_start(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return "unknown"
