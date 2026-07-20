"""Clean and normalize extracted sessions.

- Redact likely secrets/tokens in tool inputs & outputs.
- Drop empty turns (no text, no tool call, no patch).
- Truncate over-long messages.
- Optionally drop reasoning parts from training targets.
- Deduplicate identical conversations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from src.config import Config

# Patterns that look like secrets / credentials. Matched case-insensitively and
# replaced with a placeholder. Tuned to avoid nuking normal code.
_SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"), r"\1=<REDACTED>"),
    (re.compile(r"(?i)(access[_-]?token|refresh[_-]?token|bearer)\s*[:=]?\s*['\"]?[A-Za-z0-9_\-\.=]{20,}"), r"\1=<REDACTED>"),
    (re.compile(r"(?i)sk-[A-Za-z0-9]{20,}"), "<REDACTED_OPENAI>"),
    (re.compile(r"(?i)ghp_[A-Za-z0-9]{20,}"), "<REDACTED_GITHUB>"),
    (re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "<REDACTED_KEY>"),
    (re.compile(r"(?i)(password|passwd|secret)\s*[:=]\s*['\"]?.{8,}"), r"\1=<REDACTED>"),
]


def redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    return obj


def clean_message(msg: dict, cfg: Config, keep_reasoning: bool | None = None) -> dict | None:
    if keep_reasoning is None:
        keep_reasoning = cfg.get("clean", "keep_reasoning_as_context", default=False)
    max_chars = cfg.get("clean", "max_chars_per_message", default=32000)
    redact_on = cfg.get("clean", "redact_secrets", default=True)

    parts = []
    has_content = False
    for p in msg.get("parts", []):
        ptype = p.get("type")
        if ptype == "reasoning" and not keep_reasoning:
            continue
        if redact_on and ptype in ("tool", "text", "patch"):
            p = dict(p)
            if "text" in p:
                p["text"] = redact(p["text"])
            if "input" in p:
                p["input"] = _redact_obj(p["input"])
            if "output" in p:
                p["output"] = _redact_obj(p["output"])
        if "text" in p and p["text"]:
            has_content = True
        if ptype in ("tool", "patch"):
            has_content = True
        if max_chars and "text" in p and isinstance(p["text"], str) and len(p["text"]) > max_chars:
            p["text"] = p["text"][:max_chars] + "\n...[truncated]"
        parts.append(p)

    drop_empty = cfg.get("clean", "drop_empty_turns", default=True)
    if drop_empty and not has_content:
        return None

    out = dict(msg)
    out["parts"] = parts
    return out


def _conv_hash(rec: dict) -> str:
    payload = json.dumps(
        {"messages": [{"role": m.get("role"), "parts": m.get("parts")} for m in rec.get("messages", [])]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _dedup_by_session(recs: list[dict]) -> dict[str, dict]:
    """Collapse cross-source / snapshot duplicate sessions.

    The same ``session_id`` is frequently present in more than one cleaned file
    (e.g. the live opencode DB plus a NAS5 snapshot, or two Hermes cron dumps).
    They carry identical message counts but slightly divergent tool outputs. Keep
    the single most-complete copy: most messages, then largest serialized size.
    """
    best: dict[str, dict] = {}
    for r in recs:
        sid = r.get("session_id") or ""
        cur = best.get(sid)
        score = (len(r.get("messages", [])), len(json.dumps(r, sort_keys=True)))
        if cur is None:
            best[sid] = r
        else:
            cscore = (len(cur.get("messages", [])), len(json.dumps(cur, sort_keys=True)))
            if score > cscore:
                best[sid] = r
    return best


def clean_session(rec: dict, cfg: Config, keep_reasoning: bool | None = None) -> dict | None:
    cleaned_msgs = []
    for m in rec.get("messages", []):
        cm = clean_message(m, cfg, keep_reasoning=keep_reasoning)
        if cm is not None:
            cleaned_msgs.append(cm)
    if len(cleaned_msgs) < 2:
        return None
    out = dict(rec)
    out["messages"] = cleaned_msgs
    return out


def main(cfg: Config, label: str | None = None, keep_reasoning: bool = False) -> int:
    raw_dir = cfg.path("raw_dir")
    cleaned_dir = cfg.path("cleaned_dir")

    if keep_reasoning and not label:
        # Reasoning variant of Hermes: clean the flat Hermes files but keep
        # reasoning parts, writing to a dedicated labeled subdir.
        os.makedirs(cleaned_dir, exist_ok=True)
        dst = os.path.join(cleaned_dir, "hermes-reasoning")
        os.makedirs(dst, exist_ok=True)
        written = _clean_dir(raw_dir, dst, cfg, keep_reasoning=True)
        return written

    if label:
        # Clean a single labeled subdir.
        src = os.path.join(raw_dir, label)
        dst = os.path.join(cleaned_dir, label)
        os.makedirs(dst, exist_ok=True)
        return _clean_dir(src, dst, cfg, keep_reasoning=keep_reasoning)
    else:
        # Clean all subdirs.
        os.makedirs(cleaned_dir, exist_ok=True)
        total = 0
        # Hermes extraction writes flat files into raw_dir/ directly.
        if any(fn.endswith(".json") for fn in os.listdir(raw_dir)):
            total += _clean_dir(raw_dir, cleaned_dir, cfg, keep_reasoning=keep_reasoning)
        for entry in sorted(os.listdir(raw_dir)):
            src = os.path.join(raw_dir, entry)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(cleaned_dir, entry)
            os.makedirs(dst, exist_ok=True)
            total += _clean_dir(src, dst, cfg, keep_reasoning=keep_reasoning)
        return total


def _clean_dir(src: str, dst: str, cfg: Config, keep_reasoning: bool = False) -> int:
    dedupe = cfg.get("clean", "dedupe", default=True)
    seen: set[str] = set()
    written = 0
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(src, fn)) as f:
            rec = json.load(f)
        out = clean_session(rec, cfg, keep_reasoning=keep_reasoning)
        if out is None:
            continue
        if dedupe:
            h = _conv_hash(out)
            if h in seen:
                continue
            seen.add(h)
        with open(os.path.join(dst, fn), "w") as f:
            json.dump(out, f)
        written += 1
    print(f"[clean] wrote {written} cleaned sessions to {dst}")
    return written
