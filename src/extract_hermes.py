"""Harvest Hermes agent sessions into the same normalized JSON format.

Hermes exports are not yet available in this repo, so this is a stub that walks
`sources.hermes.dir` for *.json / *.jsonl session files and adapts them to the
same schema produced by `extract_opencode`. Implement the per-format adapter
when the Hermes exports land.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.config import Config


def _adapt_record(obj: dict) -> dict | None:
    # TODO: map Hermes session schema -> our normalized schema.
    # Expected output shape:
    # {"source": "hermes", "session_id": ..., "title": ..., "agent": ...,
    #  "model": ..., "messages": [{"id","role","agent","model","time","parts":[...]}]}
    if "messages" not in obj:
        return None
    return {
        "source": "hermes",
        "session_id": obj.get("id") or obj.get("session_id"),
        "title": obj.get("title", ""),
        "agent": obj.get("agent", ""),
        "model": obj.get("model", ""),
        "project_id": obj.get("project_id", ""),
        "directory": obj.get("directory", ""),
        "time_created": obj.get("time_created", 0),
        "time_updated": obj.get("time_updated", 0),
        "messages": obj.get("messages", []),
    }


def main(cfg: Config) -> int:
    h = cfg.get("sources", "hermes", default={}) or {}
    if not h.get("enabled", False):
        print("[hermes] disabled in config; skipping")
        return 0
    d = h.get("dir", "")
    if not d or not os.path.isdir(d):
        print(f"[hermes] dir not found: {d}")
        return 0
    raw_dir = cfg.path("raw_dir")
    os.makedirs(raw_dir, exist_ok=True)
    written = 0
    for fn in sorted(os.listdir(d)):
        path = os.path.join(d, fn)
        if fn.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _adapt_record(json.loads(line))
                    if rec:
                        with open(os.path.join(raw_dir, f"hermes_{rec['session_id']}.json"), "w") as o:
                            json.dump(rec, o)
                        written += 1
        elif fn.endswith(".json"):
            with open(path) as f:
                obj = json.load(f)
            # could be a list or single object
            objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                rec = _adapt_record(o)
                if rec:
                    with open(os.path.join(raw_dir, f"hermes_{rec['session_id']}.json"), "w") as out:
                        json.dump(rec, out)
                    written += 1
    print(f"[hermes] wrote {written} sessions")
    return written
