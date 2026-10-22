#!/usr/bin/env python3
"""Read-only x1 R9700 fine-tuning placement decision, never starts workloads."""
from __future__ import annotations
import argparse
import json
import subprocess
from typing import Any

TARGET = "x1-370"
PROBE = r"""
import json
from pathlib import Path
import socket
import subprocess

def cmd(args, timeout=8):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""

def proc_value(path, label):
    for line in Path(path).read_text().splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip().split()[0]
    return None

def pressure():
    values = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        row = line.split()
        if row and row[0] in ("some", "full"):
            values[row[0]] = float(dict(x.split("=", 1) for x in row[1:])["avg10"])
    return values

lease = {}
try:
    record = json.loads(Path("/home/scott/.local/state/fleet-gpu-leases/gfx1201.json").read_text())
    lease = {key: record.get(key) for key in ("device", "owner", "exclusive")}
except (OSError, ValueError):
    pass

observer = {}
try:
    # The observer correctly exits 78 on a critical GPU state while still
    # providing structured JSON on stdout. Preserve that negative evidence.
    report = subprocess.run(
        ["python3", "/home/scott/.local/bin/fleet-gpu-lease-observer", "status"],
        capture_output=True, text=True, timeout=8, check=False)
    raw = json.loads(report.stdout)
    observer = {key: raw.get(key) for key in
                ("last_observation_state", "ledger_verified", "age_seconds", "state")}
except (ValueError, TypeError):
    pass

s = {
  "host": socket.gethostname(),
  "mem_available_kib": proc_value("/proc/meminfo", "MemAvailable:"),
  "swap_free_kib": proc_value("/proc/meminfo", "SwapFree:"),
  "pressure": pressure(),
  "rocminfo_gfx1201": "gfx1201" in cmd(["rocminfo"], 8),
  "pci_amd_displays": len([x for x in cmd(["lspci", "-Dnn"]).splitlines()
                          if ("Display controller" in x or "VGA compatible controller" in x)
                          and ("AMD" in x or "Advanced Micro Devices" in x)]),
  "lease": lease,
  "observer": observer,
  "data_free_bytes": __import__("shutil").disk_usage("/media/scott/SSD_4TB").free
                      if Path("/media/scott/SSD_4TB").is_mount() else None,
}
print(json.dumps(s, sort_keys=True))
"""

def decide(data: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not isinstance(data, dict):
        return "NO_GO", ["remote_evidence_unavailable"]
    reasons = []
    if data.get("host") != TARGET:
        reasons.append("host_identity_mismatch")
    try:
        if int(data["mem_available_kib"]) < 32 * 1024 * 1024:
            reasons.append("memory_headroom_below_32GiB")
    except (TypeError, ValueError, KeyError):
        reasons.append("memory_headroom_unknown")
    try:
        if int(data["swap_free_kib"]) < 2 * 1024 * 1024:
            reasons.append("swap_headroom_below_2GiB")
    except (TypeError, ValueError, KeyError):
        reasons.append("swap_headroom_unknown")
    try:
        p = data["pressure"]
        if not (0 <= float(p["some"]) <= 2 and 0 <= float(p["full"]) <= 1):
            reasons.append("memory_pressure_above_gate")
    except (KeyError, ValueError, TypeError):
        reasons.append("memory_pressure_unknown")
    displays = data.get("pci_amd_displays")
    if (data.get("rocminfo_gfx1201") is not True
            or not isinstance(displays, int) or isinstance(displays, bool) or displays < 2):
        reasons.append("r9700_physical_gpu_missing")
    lease = data.get("lease")
    if not isinstance(lease, dict):
        lease = {}
    if (lease.get("device") != "gfx1201" or lease.get("owner") != "predictive-finetuning"
            or lease.get("exclusive") is not True):
        reasons.append("r9700_lease_not_verified")
    ob = data.get("observer")
    if not isinstance(ob, dict):
        ob = {}
    if ob.get("ledger_verified") is not True or ob.get("last_observation_state") != "healthy":
        reasons.append("gpu_observer_not_healthy")
    try:
        age = float(ob["age_seconds"])
        if not 0 <= age <= 45 * 60:
            reasons.append("gpu_observer_stale")
    except (KeyError, TypeError, ValueError):
        reasons.append("gpu_observer_stale")
    try:
        if int(data["data_free_bytes"]) < 200 * 1024 ** 3:
            reasons.append("local_data_space_below_200GiB")
    except (TypeError, ValueError, KeyError):
        reasons.append("local_data_space_unknown")
    return ("NO_GO", reasons) if reasons else ("PREFLIGHT_PASS", [])

def collect(target: str) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
             "-o", "ConnectTimeout=6", target, "python3 -"],
            input=PROBE, text=True, capture_output=True, timeout=24, check=False)
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout)
        return result if isinstance(result, dict) else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=(TARGET,), default=TARGET)
    args = ap.parse_args()
    evidence = collect(args.target)
    result, reasons = decide(evidence)
    print(json.dumps({"target": args.target, "status": result, "may_launch_training": False,
                      "read_only": True, "reasons": reasons, "evidence": evidence}, sort_keys=True))
    # PREFLIGHT_PASS is NOT dispatch authority, and never returns success for automation.
    return 78 if result == "NO_GO" else 0

if __name__ == "__main__":
    raise SystemExit(main())
