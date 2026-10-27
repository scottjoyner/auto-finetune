"""Lan lm-fleet-router helper for the benchmark's large-reference runner.

The local 12GB iGPU only fits the 3B models we finetune. For a "much larger
foundational model" reference, we hit the lan fleet router, which exposes
OpenAI-compatible /v1 endpoints on each node (see ~/.config/opencode/endpoints.json).

This module is READ-ONLY: it parses endpoints.json and returns candidate
(base_url, model) pairs so the bench harness can drive a big model via the
normal `api` runner. No credentials are read or stored here.

Hardening (2026-10-27):
  - Input validation on all public functions.
  - Atomic file read with retry on transient I/O errors.
  - Defensive parsing of malformed endpoint entries.
  - Timeout-aware HTTP health probing (optional).
  - Deduplication and sorting for deterministic output.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

DEFAULT_ENDPOINTS = "/home/scott/.config/opencode/endpoints.json"

# models we treat as "large reference" candidates (>= ~12B effective)
_LARGE_HINTS = ("35b", "12b", "20b", "27b", "32b", "70b", "gpt-oss", "opus", "claude")

# File-read retry parameters
_READ_RETRIES = 3
_READ_RETRY_DELAY = 0.1

# Health-probe timeout (seconds)
_PROBE_TIMEOUT = 5.0


def _validate_path(path: str) -> str:
    """Validate and normalize an endpoints path."""
    if not path or not isinstance(path, str):
        raise ValueError("endpoints_path must be a non-empty string")
    return str(Path(path).expanduser())


def _atomic_read_json(path: str, retries: int = _READ_RETRIES) -> dict:
    """Read and parse JSON with retry on transient I/O errors.

    Returns {} on any failure (file missing, corrupt JSON, permission error)
    so callers never crash on a missing or unreadable endpoints file.
    """
    p = Path(path)
    for attempt in range(max(1, retries)):
        try:
            if not p.exists():
                return {}
            text = p.read_text(encoding="utf-8")
            if not text.strip():
                return {}
            data = json.loads(text)
            if not isinstance(data, dict):
                return {}
            return data
        except (OSError, IOError, PermissionError) as exc:
            if attempt < retries - 1:
                time.sleep(_READ_RETRY_DELAY * (attempt + 1))
                continue
            return {}
        except (json.JSONDecodeError, ValueError):
            return {}
        except Exception:
            return {}
    return {}


def load_fleet(endpoints_path: str = DEFAULT_ENDPOINTS) -> dict:
    """Load and validate the fleet endpoints file.

    Returns {} on any error (missing file, corrupt JSON, permission denied).
    """
    path = _validate_path(endpoints_path)
    return _atomic_read_json(path)


def _validate_node(node: object) -> bool:
    """Check that a node entry has the minimum required fields."""
    if not isinstance(node, dict):
        return False
    if not node.get("online"):
        return False
    base = node.get("base_url")
    if not base or not isinstance(base, str):
        return False
    return True


def _normalize_model_name(model: object) -> str | None:
    """Normalize a model entry to a string or None if invalid."""
    if isinstance(model, str) and model.strip():
        return model.strip()
    if isinstance(model, dict):
        name = model.get("name") or model.get("id") or model.get("model")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def list_models(endpoints_path: str = DEFAULT_ENDPOINTS,
                only_large: bool = True) -> list[dict]:
    """Return a flat list of {"node", "base_url", "model"} for online nodes.

    If only_large, keep only models whose name hints at a large size.
    Results are deduplicated and sorted by (node, model) for determinism.
    """
    fleet = load_fleet(endpoints_path)
    nodes = fleet.get("nodes", [])
    if not isinstance(nodes, list):
        return []

    out = []
    for node in nodes:
        if not _validate_node(node):
            continue
        base = node["base_url"].rstrip("/")
        node_name = str(node.get("name", "?")).strip() or "?"

        # Collect models from both available and loaded lists.
        model_sources = []
        for key in ("available_models", "loaded_models"):
            val = node.get(key, [])
            if isinstance(val, list):
                model_sources.extend(val)

        for m in model_sources:
            name = _normalize_model_name(m)
            if name is None:
                continue
            if only_large and not any(h in name.lower() for h in _LARGE_HINTS):
                continue
            out.append({"node": node_name, "base_url": base, "model": name})

    # de-dup by (base_url, model)
    seen = set()
    uniq = []
    for r in out:
        key = (r["base_url"], r["model"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    # Sort for deterministic output
    uniq.sort(key=lambda r: (r["node"], r["model"]))
    return uniq


def pick_model(hint: Optional[str] = None,
               endpoints_path: str = DEFAULT_ENDPOINTS) -> Optional[dict]:
    """Pick a large reference model, optionally filtering by a name substring.

    Returns None if no models are available or hint matches nothing.
    """
    models = list_models(endpoints_path)
    if not models:
        return None
    if hint and isinstance(hint, str) and hint.strip():
        hint_lower = hint.strip().lower()
        for m in models:
            if hint_lower in m["model"].lower():
                return m
        # Hint didn't match; return None rather than a wrong model
        return None
    return models[0]


def probe_health(base_url: str, timeout: float = _PROBE_TIMEOUT) -> bool:
    """Probe whether a fleet endpoint is healthy.

    Returns True if the endpoint responds with HTTP 200 within timeout.
    Returns False on any error (timeout, connection refused, etc.).
    """
    if not base_url or not isinstance(base_url, str):
        return False
    try:
        import urllib.request
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def filter_healthy(models: list[dict],
                   timeout: float = _PROBE_TIMEOUT) -> list[dict]:
    """Filter a model list to only those whose endpoint is healthy."""
    if not isinstance(models, list):
        return []
    healthy = []
    for m in models:
        if not isinstance(m, dict):
            continue
        base = m.get("base_url")
        if base and probe_health(base, timeout):
            healthy.append(m)
    return healthy
