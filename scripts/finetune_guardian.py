#!/usr/bin/env python3
"""xwing guardian: hold-aware trainer transitions and temperature, no job mutation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, build_opener, ProxyHandler

STOP = Path("/media/scott/data/finetune-staging/launch-next.stop")
STATE = Path.home() / ".local/state/auto-finetune/guardian-state.json"
LOG = Path.home() / "finetune-guardian.log"
SIGNAL = "http://127.0.0.1:18080/api/v1/rpc"
HEAT_C = 88


def is_training_command(args: list[str]) -> bool:
    """Count exact Python training entries, not shell/grep/harness ancestry."""
    if not args or not args[0]:
        return False
    exe = Path(args[0]).name.lower()
    if not (exe.startswith("python") or exe.startswith("pypy")):
        return False
    try:
        idx = args.index("-m")
    except ValueError:
        return False
    if idx + 1 >= len(args):
        return False
    module = args[idx + 1]
    if module in ("src.train", "src.dpo"):
        return True
    if module == "src.cli" and idx + 2 < len(args):
        return args[idx + 2] in ("train", "dpo")
    return False


def count_trainers(proc: Path = Path("/proc")) -> int:
    count = 0
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            if is_training_command([p.decode("utf-8", "replace")
                                    for p in path.read_bytes().split(b"\0") if p]):
                count += 1
        except OSError:
            continue
    return count


def temperature() -> int | None:
    try:
        result = subprocess.run(["rocm-smi", "--showtemp", "--csv"], capture_output=True,
                                text=True, timeout=6, check=False)
        if result.returncode != 0:
            return None
        rows = result.stdout.splitlines()
        if len(rows) < 2 or "Temperature" not in rows[0]:
            return None
        samples = []
        for line in rows[1:]:
            if not line.startswith("card"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", parts[1])
            if match:
                temp = float(match.group(1))
                if 0 <= temp <= 120:
                    samples.append(int(temp))
        return max(samples) if samples else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def decide(previous: dict, stopped: bool, trainers: int, temp_c: int | None) -> tuple[dict, list[str]]:
    """Pure policy: no death alarm from a held/never-observed training run."""
    if not isinstance(previous, dict):
        previous = {}
    prev_active = previous.get("trainers_active") is True
    old_alerts = previous.get("latched", [])
    if not isinstance(old_alerts, list):
        old_alerts = []
    held = bool(stopped)
    running = trainers > 0
    streak = int(previous.get("hot_streak", 0)) if type(previous.get("hot_streak", 0)) is int else 0
    streak = min(streak + 1, 2) if temp_c is not None and temp_c >= HEAT_C else 0
    conditions: list[str] = []
    if held and running:
        conditions.append("trainer_running_during_hold")
    if not held and prev_active and not running:
        conditions.append("trainer_disappeared")
    if streak >= 2:
        conditions.append("sustained_temperature")
    new_events = [event for event in conditions if event not in old_alerts]
    # Transition-to-absence is one-shot, even once next sample is inactive.
    state = {
        "trainers_active": running,
        "held": held,
        "hot_streak": streak,
        "latched": conditions,
    }
    return state, new_events


def _read_state() -> dict:
    try:
        item = json.loads(STATE.read_text())
        return item if isinstance(item, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_state(data: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE.parent, 0o700)
    temporary = STATE.with_name(STATE.name + ".tmp." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(data, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, STATE)


def _append_log(entry: dict) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _send_event(event: str, ts: str, temp_c: int | None) -> bool:
    descriptions = {
        "trainer_disappeared": "Trainer process disappeared after previously running.",
        "trainer_running_during_hold": "Trainer detected while recovery stop-file is present.",
        "sustained_temperature": "GPU temperature >=88C on two consecutive checks.",
    }
    message = "xwing finetune GUARDIAN: " + descriptions[event] + " Event=" + event + " time=" + ts
    if temp_c is not None:
        message += " gpu_temp_c=" + str(temp_c)
    # Construct JSON safely and never expose model traces, tokens or full argv.
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "send",
                          "params": {"noteToSelf": True, "message": message}}).encode()
    request = Request(SIGNAL, data=payload, method="POST",
                      headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=7) as response:
            return response.status == 200
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="read only: no state, log or notification writes")
    args = parser.parse_args()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    observed = count_trainers()
    temp_c = temperature()
    held = STOP.exists()
    before = _read_state()
    after, new_events = decide(before, held, observed, temp_c)
    record = {"timestamp": ts, "hold": held, "trainers": observed,
              "temperature_c": temp_c, "alerts": new_events, "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(record, sort_keys=True))
        return 0
    # Send only transitions, no repeated alerts on every five-minute tick.
    # Persist the evaluated state; failure to send is logged for diagnosis.
    results = {event: _send_event(event, ts, temp_c) for event in new_events}
    after["last_alert_outcomes"] = results
    _atomic_state(after)
    _append_log(record | {"notification_outcomes": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
