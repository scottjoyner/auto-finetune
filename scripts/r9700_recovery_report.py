#!/usr/bin/env python3
"""Read-only x1 R9700 recovery diagnostics. NEVER enables or launches training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts.fleet_training_placement import collect, decide

GOVEE_BASE = "https://developer-api.govee.com/v1"
PLUG_C = "06:F0:98:17:3C:6A:AB:D0"
KEY_PATH = Path.home() / ".config/govee/api-key"


class _NoRedirect(HTTPRedirectHandler):
    # Never forward Govee API credentials to a redirected hostname.
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("Govee redirect refused")


def _govee_get(route: str, api_key: str) -> dict:
    """GET-only and allowlisted routes; never log credentials or API bodies."""
    if route != "/devices" and not route.startswith("/devices/state?"):
        raise ValueError("non-read-only route rejected")
    request = Request(
        GOVEE_BASE + route,
        headers={"Govee-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )
    with build_opener(_NoRedirect()).open(request, timeout=7) as response:
        if response.status != 200:
            raise ValueError("Govee HTTP failure")
        data = response.read(128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("Govee response too large")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("invalid Govee response")
    return result


def read_relay() -> dict:
    """No secret bytes, device lists, or private provider messages in result."""
    try:
        mode = stat.S_IMODE(KEY_PATH.stat().st_mode)
        if mode & 0o077:
            return {"state": "UNKNOWN", "reason": "credential_permissions_unsafe"}
        key = KEY_PATH.read_text().strip()
        if not key:
            return {"state": "UNKNOWN", "reason": "credential_unavailable"}
        devices = _govee_get("/devices", key)["data"]["devices"]
        matching = [
            d for d in devices
            if isinstance(d, dict) and str(d.get("device", "")).upper() == PLUG_C
        ]
        if len(matching) != 1:
            return {"state": "UNKNOWN", "reason": "plug_c_identity_not_unique"}
        device = matching[0]
        query = urlencode({"device": PLUG_C, "model": device["model"]})
        status = _govee_get("/devices/state?" + query, key)["data"]["properties"]
        properties = {k: v for entry in status for k, v in entry.items()}
        power = properties.get("powerState")
        online = properties.get("online")
        if power not in ("on", "off") or not isinstance(online, bool):
            return {"state": "UNKNOWN", "reason": "relay_state_ambiguous"}
        return {"state": power.upper(), "online": online, "reason": "provider_relay_state_only"}
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return {"state": "UNKNOWN", "reason": "relay_telemetry_unavailable"}


def assess(relay: dict, placement: dict | None) -> dict:
    """Pure conservative policy: even both green never authorize training."""
    pstatus, reasons = decide(placement)
    result_reasons = list(reasons)
    if relay.get("state") != "ON" or relay.get("online") is not True:
        result_reasons.append("plug_c_relay_not_confirmed_on")
    operator_review = not result_reasons and pstatus == "PREFLIGHT_PASS"
    # An upstream relay cannot prove dock PSU, card, auxiliary PCIe power,
    # or cable continuity; this command never grants training authority.
    return {
        "status": "PHYSICAL_REVIEW_REQUIRED" if operator_review else "NO_GO",
        "may_launch_training": False,
        "no_power_control_commands": True,
        "relay": relay,
        "reasons": result_reasons,
        "placement": {"status": pstatus, "reasons": reasons, "evidence": placement},
        "operator_review_only": operator_review,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["x1-370"], default="x1-370")
    args = parser.parse_args()
    relay = read_relay()
    placement = collect(args.target)
    report = assess(relay, placement)
    report["target"] = args.target
    print(json.dumps(report, sort_keys=True))
    return 0 if report["operator_review_only"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
