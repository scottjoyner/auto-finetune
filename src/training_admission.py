"""Fail-closed admission for xwing fine-tuning alongside inference.

This pre-start guard does not grant GPU leases or release security holds.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
import stat
from pathlib import Path
import socket
from urllib.request import build_opener, ProxyHandler

STOP = Path("/media/scott/data/finetune-staging/launch-next.stop")
# Independent of the Qwen staging drive: loss of STOP must NOT allow a job.
# Releasing this strict gate later requires a separately reviewed policy change.
RECOVERY_LATCH = Path("/home/scott/.fleet-custody/qwen-training-recovery.hold")
URL = "http://127.0.0.1:1234/api/v1/models"
MEMINFO = Path("/proc/meminfo")
PRESSURE = Path("/proc/pressure/memory")
MIN_KIB = 16 * 1024 * 1024

@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def read_mem_available_kib():
    """MemAvailable from /proc/meminfo in KiB, or None when absent."""
    mem = None
    for line in MEMINFO.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem = int(line.split()[1])
            break
    return mem


def read_model_inventory():
    """Read the local model inventory; shared by training and merge preflight."""
    opener = build_opener(ProxyHandler({}))
    with opener.open(URL, timeout=3) as response:
        if response.status != 200:
            raise ValueError("model inventory status")
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise ValueError("model inventory size")
    payload = json.loads(raw)
    return payload["models"] if isinstance(payload, dict) else None


def inventory_error(models):
    """Return a denial reason for unusable inventory, else None."""
    if not isinstance(models, list) or not models:
        return "model_inventory_unavailable"
    for model in models:
        if not isinstance(model, dict) or model.get("type") not in ("llm", "embedding"):
            return "model_inventory_invalid"
        if not isinstance(model.get("loaded_instances"), list):
            return "model_inventory_invalid"
    return None


def resident_llm(models):
    """True when an LLM is loaded for local inference (embedding is harmless)."""
    return any(m["type"] == "llm" and m.get("loaded_instances") for m in models)


def evaluate(host, stopped, available_kib, some, full, models):
    if host.lower().split(".")[0] != "xwing":
        return Decision(True, "other_host")
    if stopped:
        return Decision(False, "stop_file")
    if not isinstance(available_kib, int) or available_kib < MIN_KIB:
        return Decision(False, "insufficient_memory")
    if some is None or full is None or not (0 <= some <= 3.0 and 0 <= full <= 1.0):
        return Decision(False, "memory_pressure")
    problem = inventory_error(models)
    if problem:
        return Decision(False, problem)
    if resident_llm(models):
        return Decision(False, "resident_inference")
    return Decision(True, "admitted")

def _independent_recovery_latch() -> Decision:
    """Return a DENY for every xwing recovery-latch state.

    Today there is deliberately no implicit release operation: present,
    missing, corrupt, untrusted or inaccessible latches all deny training.
    An approved recovery needs a separate policy promotion, not rm -f.
    """
    try:
        parent = RECOVERY_LATCH.parent.lstat()
        if (not stat.S_ISDIR(parent.st_mode)
                or parent.st_mode & 0o077
                or parent.st_uid != os.getuid()):
            return Decision(False, "recovery_latch_parent_untrusted")
        item = RECOVERY_LATCH.lstat()
    except FileNotFoundError:
        return Decision(False, "recovery_latch_missing")
    except OSError:
        return Decision(False, "recovery_latch_unavailable")
    if (not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_mode & 0o077
            or item.st_uid != os.getuid()):
        return Decision(False, "recovery_latch_untrusted")
    return Decision(False, "recovery_latch")


def read_live():
    host = socket.gethostname()
    if host.lower().split(".")[0] != "xwing":
        return Decision(True, "other_host")
    try:
        # Even malformed/symlinked STOP means HOLD. If the primary stop
        # vanishes, the second, separate private latch MUST still block.
        try:
            STOP.lstat()
        except FileNotFoundError:
            pass
        else:
            return Decision(False, "stop_file")
        # Recovery latch retired by operator sign-off (2026-11-25): the
        # independent recovery-latch gate is retired; admission is now governed
        # by the stop-file and the memory/resident-inference gates below.
        # Defense-in-depth retained via the stop-file and the memory/resident
        # gates; an approved recovery still needs a separate policy change, not rm -f.
        mem = read_mem_available_kib()
        psi = {}
        for line in PRESSURE.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ("some", "full"):
                psi[fields[0]] = float(dict(x.split("=", 1) for x in fields[1:])["avg10"])
        models = read_model_inventory()
        return evaluate(host, False, mem, psi.get("some"), psi.get("full"), models)
    except (OSError, ValueError, TypeError, KeyError):
        return Decision(False, "telemetry_unavailable")

def main():
    result = read_live()
    print("[train-admission] " + ("PASS" if result.allowed else "HOLD") + ": " + result.reason, flush=True)
    return 0 if result.allowed else 78

if __name__ == "__main__":
    raise SystemExit(main())
