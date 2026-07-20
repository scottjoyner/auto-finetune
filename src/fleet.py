"""Lan lm-fleet-router helper for the benchmark's large-reference runner.

The local 12GB iGPU only fits the 3B models we finetune. For a "much larger
foundational model" reference, we hit the lan fleet router, which exposes
OpenAI-compatible /v1 endpoints on each node (see ~/.config/opencode/endpoints.json).

This module is READ-ONLY: it parses endpoints.json and returns candidate
(base_url, model) pairs so the bench harness can drive a big model via the
normal `api` runner. No credentials are read or stored here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_ENDPOINTS = "/home/scott/.config/opencode/endpoints.json"

# models we treat as "large reference" candidates (>= ~12B effective)
_LARGE_HINTS = ("35b", "12b", "20b", "27b", "32b", "70b", "gpt-oss", "opus", "claude")


def load_fleet(endpoints_path: str = DEFAULT_ENDPOINTS) -> dict:
    p = Path(endpoints_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def list_models(endpoints_path: str = DEFAULT_ENDPOINTS,
                only_large: bool = True) -> list[dict]:
    """Return a flat list of {"node", "base_url", "model"} for online nodes.

    If only_large, keep only models whose name hints at a large size.
    """
    fleet = load_fleet(endpoints_path)
    out = []
    for node in fleet.get("nodes", []):
        if not node.get("online"):
            continue
        base = node.get("base_url")
        if not base:
            continue
        for m in node.get("available_models", []) + node.get("loaded_models", []):
            if only_large and not any(h in m.lower() for h in _LARGE_HINTS):
                continue
            out.append({"node": node.get("name", "?"), "base_url": base, "model": m})
    # de-dup by (base_url, model)
    seen = set()
    uniq = []
    for r in out:
        key = (r["base_url"], r["model"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def pick_model(hint: Optional[str] = None,
               endpoints_path: str = DEFAULT_ENDPOINTS) -> Optional[dict]:
    """Pick a large reference model, optionally filtering by a name substring."""
    models = list_models(endpoints_path)
    if not models:
        return None
    if hint:
        for m in models:
            if hint.lower() in m["model"].lower():
                return m
    return models[0]
