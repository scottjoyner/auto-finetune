"""Fail-closed admission for xwing fine-tuning alongside inference.

This pre-start guard does not grant GPU leases or release security holds.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import socket
from urllib.request import build_opener, ProxyHandler

STOP = Path("/media/scott/data/finetune-staging/launch-next.stop")
URL = "http://127.0.0.1:1234/api/v1/models"
MIN_KIB = 16 * 1024 * 1024

@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str

def evaluate(host, stopped, available_kib, some, full, models):
    if host.lower().split(".")[0] != "xwing":
        return Decision(True, "other_host")
    if stopped:
        return Decision(False, "stop_file")
    if not isinstance(available_kib, int) or available_kib < MIN_KIB:
        return Decision(False, "insufficient_memory")
    if some is None or full is None or not (0 <= some <= 3.0 and 0 <= full <= 1.0):
        return Decision(False, "memory_pressure")
    if not isinstance(models, list) or not models:
        return Decision(False, "model_inventory_unavailable")
    for model in models:
        if not isinstance(model, dict) or model.get("type") not in ("llm", "embedding"):
            return Decision(False, "model_inventory_invalid")
        instances = model.get("loaded_instances")
        if not isinstance(instances, list):
            return Decision(False, "model_inventory_invalid")
        if model["type"] == "llm" and instances:
            return Decision(False, "resident_inference")
    return Decision(True, "admitted")

def read_live():
    host = socket.gethostname()
    if host.lower().split(".")[0] != "xwing":
        return Decision(True, "other_host")
    try:
        if STOP.exists():
            return Decision(False, "stop_file")
        mem = None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem = int(line.split()[1])
                break
        psi = {}
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ("some", "full"):
                psi[fields[0]] = float(dict(x.split("=", 1) for x in fields[1:])["avg10"])
        opener = build_opener(ProxyHandler({}))
        with opener.open(URL, timeout=3) as response:
            if response.status != 200:
                raise ValueError("model inventory status")
            raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("model inventory size")
        payload = json.loads(raw)
        models = payload["models"] if isinstance(payload, dict) else None
        return evaluate(host, False, mem, psi.get("some"), psi.get("full"), models)
    except (OSError, ValueError, TypeError, KeyError):
        return Decision(False, "telemetry_unavailable")

def main():
    result = read_live()
    print("[train-admission] " + ("PASS" if result.allowed else "HOLD") + ": " + result.reason, flush=True)
    return 0 if result.allowed else 78

if __name__ == "__main__":
    raise SystemExit(main())
