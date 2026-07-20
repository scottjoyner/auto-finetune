"""Reconstruct cleaned sessions into training examples.

Each cleaned session is turned into one (or several, if windowed) conversation
following a chat template: chatml | alpaca | sharegpt | hermes.

Hermes format uses the tokenizer's chat_template with proper tool_calls/tool roles.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.clean import _dedup_by_session
from src.config import Config


def _render_part(p: dict) -> str:
    t = p.get("type")
    if t == "text":
        return p.get("text", "")
    if t == "tool":
        tool = p.get("tool", "tool")
        call_id = p.get("call_id", "")
        inp = p.get("input")
        out = p.get("output")
        lines = [f"<tool_call name=\"{tool}\" call_id=\"{call_id}\">"]
        lines.append(json.dumps(inp, indent=2) if isinstance(inp, (dict, list)) else str(inp or ""))
        lines.append("\\u276E\\u276E\\u276E")
        if out:
            lines.append("<tool_result>")
            lines.append(out if isinstance(out, str) else json.dumps(out, indent=2))
            lines.append("</tool_result>")
        return "\n".join(lines)
    if t == "patch":
        files = p.get("files") or []
        return "<patch files=\"" + ", ".join(files) + "\">\n(diff applied)\n</patch>"
    if t == "reasoning":
        return p.get("text", "")
    return ""


def _render_message(m: dict) -> str:
    return "\n".join(_render_part(p) for p in m.get("parts", []) if _render_part(p)).strip()


def _extract_tool_calls(m: dict) -> list[dict]:
    tool_calls = []
    for p in m.get("parts", []):
        if p.get("type") == "tool":
            tool_calls.append({
                "id": p.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": p.get("tool", "unknown"),
                    "arguments": json.dumps(p.get("input", {}))
                }
            })
    return tool_calls


def _extract_tool_results(m: dict) -> list[dict]:
    results = []
    for p in m.get("parts", []):
        if p.get("type") == "tool_result" or (p.get("type") == "tool" and p.get("output") is not None):
            results.append({
                "role": "tool",
                "content": p.get("output", "") if isinstance(p.get("output"), str) else json.dumps(p.get("output", "")),
                "tool_call_id": p.get("call_id", "")
            })
    return results


def to_chatml(messages: list[dict], system: str) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = _render_message(m)
        if not content:
            continue
        role = "assistant" if m.get("role") == "assistant" else "user"
        out.append({"role": role, "content": content})
    return out


def to_sharegpt(messages: list[dict], system: str) -> list[dict]:
    conv = []
    for m in messages:
        content = _render_message(m)
        if not content:
            continue
        role = "gpt" if m.get("role") == "assistant" else "human"
        conv.append({"from": role, "value": content})
    if system:
        conv.insert(0, {"from": "system", "value": system})
    return conv


def to_alpaca(messages: list[dict], system: str) -> dict:
    user_turns, asst_turns = [], []
    for m in messages:
        c = _render_message(m)
        if not c:
            continue
        if m.get("role") == "assistant":
            asst_turns.append(c)
        else:
            user_turns.append(c)
    instruction = (system + "\n\n" if system else "") + (user_turns[0] if user_turns else "")
    input_text = "\n".join(user_turns[1:]) if len(user_turns) > 1 else ""
    output_text = asst_turns[-1] if asst_turns else ""
    return {"instruction": instruction, "input": input_text, "output": output_text}


def to_hermes(messages: list[dict], system: str) -> dict:
    out = []
    sys_content = system if system else "You are a helpful assistant."
    out.append({"role": "system", "content": sys_content})

    for m in messages:
        role = m.get("role")
        if role == "assistant":
            content = _render_message(m)
            tool_calls = _extract_tool_calls(m)
            if tool_calls:
                out.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            else:
                out.append({"role": "assistant", "content": content})
        elif role == "user":
            content = _render_message(m)
            if content:
                out.append({"role": "user", "content": content})
        elif role == "tool":
            for tr in _extract_tool_results(m):
                out.append(tr)

    return {"messages": out}


def main(cfg: Config, source: str | None = None, label: str | None = None) -> int:
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")
    os.makedirs(dataset_dir, exist_ok=True)

    template = cfg.get("format", "template", default="chatml")
    system = cfg.get("format", "system_prompt", default="") or ""
    max_turns = cfg.get("format", "max_turns_per_example", default=0) or 0
    max_chars = cfg.get("format", "max_chars_per_example", default=24000) or 0

    def _format_one(src_dir: str, out_path: str, filter_source: str | None) -> int:
        examples: list[Any] = []
        sources_seen: set[str] = set()
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(src_dir, fn)) as f:
                rec = json.load(f)
            src = rec.get("source", "")
            sources_seen.add(src)
            if filter_source and src != filter_source:
                continue
            msgs = rec.get("messages", [])
            windows = _window_messages(msgs, max_turns, max_chars)
            for w in windows:
                if len(w) < 2:
                    continue
                examples.append(_format_window(w, template, system))
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        return len(examples)

    def _collect_examples(src_dir: str, max_turns: int, max_chars: int,
                          template: str, system: str) -> list[Any]:
        examples: list[Any] = []
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(src_dir, fn)) as f:
                rec = json.load(f)
            if rec.get("source") != "opencode":
                continue
            msgs = rec.get("messages", [])
            windows = _window_messages(msgs, max_turns, max_chars)
            for w in windows:
                if len(w) < 2:
                    continue
                examples.append(_format_window(w, template, system))
        return examples

    if label == "opencode-all":
        # Merge every opencode source subdir (ssd, nas5-*, opencode-<project>)
        # into one corpus. Identified by peeking each cleaned subdir's records.
        out_path = os.path.join(dataset_dir, "train.opencode-all.jsonl")
        examples: list[Any] = []
        for entry in sorted(os.listdir(cleaned_dir)):
            src_dir = os.path.join(cleaned_dir, entry)
            if not os.path.isdir(src_dir):
                continue
            is_opencode = False
            for fn in sorted(os.listdir(src_dir)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(src_dir, fn)) as f:
                        rec = json.load(f)
                    if rec.get("source") == "opencode":
                        is_opencode = True
                except Exception:
                    pass
                break
            if not is_opencode:
                continue
            examples.extend(_collect_examples(src_dir, max_turns, max_chars, template, system))
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[format] opencode-all: {len(examples)} examples -> {out_path}")
        return len(examples)
    if label:
        # Format a single labeled cleaned subdir.
        src_dir = os.path.join(cleaned_dir, label)
        suffix = f".{label}"
        if source:
            suffix += f".{source}"
        out_path = os.path.join(dataset_dir, f"train{suffix}.jsonl")
        n = _format_one(src_dir, out_path, source)
        print(f"[format] {label} ({source or 'all'}): {n} examples -> {out_path}")
        return n
    else:
        # Hermes cleaning writes flat files into cleaned_dir/ directly, plus
        # opencode sources live in per-source subdirs. Emit one train.<label>.jsonl
        # per subdir, a merged train.jsonl, and (when --source is given) a
        # train.<source>.jsonl. Each output file is written exactly once.
        total = 0
        for entry in sorted(os.listdir(cleaned_dir)):
            src_dir = os.path.join(cleaned_dir, entry)
            if not os.path.isdir(src_dir):
                continue
            out_path = os.path.join(dataset_dir, f"train.{entry}.jsonl")
            n = _format_one(src_dir, out_path, None)  # always full for per-label
            print(f"[format] {entry}: {n} examples -> {out_path}")
            total += n
        # Merged (optionally source-filtered) train.jsonl.
        out_path = os.path.join(dataset_dir, "train.jsonl")
        n = _format_one(cleaned_dir, out_path, source)
        label_str = source or "merged"
        print(f"[format] {label_str}: {n} examples -> {out_path}")
        total += n
        # Per-source file only when a --source filter is active (otherwise it
        # would duplicate the merged file written just above).
        if source:
            out_path = os.path.join(dataset_dir, f"train.{source}.jsonl")
            n = _format_one(cleaned_dir, out_path, source)
            print(f"[format] {source}: {n} examples -> {out_path}")
            total += n
        return total


def _window_messages(msgs: list[dict], max_turns: int, max_chars: int) -> list[list[dict]]:
    if max_turns and len(msgs) > max_turns:
        out = []
        step = max(1, max_turns // 2)
        for i in range(0, len(msgs), step):
            out.append(msgs[i:i + max_turns])
        return out
    if max_chars:
        out = []
        start = 0
        while start < len(msgs):
            end = start
            total = 0
            while end < len(msgs):
                est = sum(len(_render_part(p)) for p in msgs[end].get("parts", []))
                if total + est > max_chars and end > start:
                    break
                total += est
                end += 1
            out.append(msgs[start:end])
            if end >= len(msgs):
                break
            start = max(start + 1, end - 1)
        return out
    return [msgs]


def _format_window(msgs: list[dict], template: str, system: str) -> Any:
    if template == "alpaca":
        return to_alpaca(msgs, system)
    if template == "sharegpt":
        return {"conversations": to_sharegpt(msgs, system)}
    if template == "hermes":
        return to_hermes(msgs, system)
    return {"messages": to_chatml(msgs, system)}


# Merged-corpus label set: every per-source training dataset EXCEPT the
# already-merged opencode-all (which is a subset of these) and any eval split.
_COMBINE_LABELS = (
    "ssd",
    "nas5-main",
    "nas5-20260717",
    "nas5-old-broken",
    "nas5-recover-old",
    "opencode-portfolio",
    "hermes-reasoning",
)


def combine(cfg: Config) -> int:
    """Merge all per-source datasets into one train.combined.jsonl.

    This is the "finetune them all together" equivalent: a single corpus over
    the union of sources, ready for one LoRA run with --label=combined.
    """
    dataset_dir = cfg.path("dataset_dir")
    out_path = os.path.join(dataset_dir, "train.combined.jsonl")
    seen: set[str] = set()
    total = 0
    with open(out_path, "w") as out:
        for label in _COMBINE_LABELS:
            src = os.path.join(dataset_dir, f"train.{label}.jsonl")
            if not os.path.exists(src):
                print(f"[combine] skip missing {src}")
                continue
            n = 0
            for line in open(src):
                line = line.strip()
                if not line:
                    continue
                # de-dupe identical examples across sources
                if line in seen:
                    continue
                seen.add(line)
                out.write(line + "\n")
                n += 1
            print(f"[combine] {label}: {n} examples")
            total += n
    print(f"[combine] wrote {total} unique examples -> {out_path}")
    return total


def iter_cleaned_records(cleaned_dir: str) -> list[dict]:
    """Yield every cleaned session record under ``cleaned_dir`` (flat + subdirs)."""
    recs: list[dict] = []
    for path in sorted(Path(cleaned_dir).rglob("*.json")):
        try:
            recs.append(json.loads(path.read_text()))
        except Exception:
            continue
    return recs


def emit_strata(cfg: "Config", bucket_map: dict, out_dir: str,
                balance: bool = False, cap: int | None = None) -> dict:
    """Emit one training jsonl per task-bucket into ``out_dir`` (staging).

    Reads the (deduplicated) cleaned corpus, looks up each session's bucket from
    ``bucket_map`` (session_id -> {bucket, ...}; falls back to a live ``analyze``
    classification when a session is absent), and writes ``train.<bucket>.jsonl``
    into ``out_dir`` using the configured chat template / windowing.

    When ``balance`` is set, every bucket is upsampled by repetition to ``cap``
    examples (default: the largest bucket's count) and a combined
    ``train.balanced.jsonl`` is also written — this directly addresses the
    actionable buckets being under-represented in the merged corpus.

    ``out_dir`` must be a staging path (e.g. ``<data>/analysis``) — never the
    live ``datasets/`` dir, which a running training job memory-maps.
    """
    os.makedirs(out_dir, exist_ok=True)
    template = cfg.get("format", "template", default="chatml")
    system = cfg.get("format", "system_prompt", default="") or ""
    max_turns = cfg.get("format", "max_turns_per_example", default=0) or 0
    max_chars = cfg.get("format", "max_chars_per_example", default=24000) or 0

    unique = _dedup_by_session(iter_cleaned_records(cfg.path("cleaned_dir")))
    buckets: dict[str, list] = defaultdict(list)
    for sid, rec in unique.items():
        b = (bucket_map.get(sid) or {}).get("bucket")
        if not b:
            from src import analyze as _a
            b = _a.classify_bucket(_a.extract_features(rec))
        for w in _window_messages(rec.get("messages", []), max_turns, max_chars):
            if len(w) < 2:
                continue
            buckets[b].append(_format_window(w, template, system))

    target = cap if cap else (max(len(v) for v in buckets.values()) if buckets else 0)
    counts: dict[str, int] = {}
    for b, exs in sorted(buckets.items()):
        if balance and target:
            if len(exs) < target:
                # upsample by repetition
                reps = target // len(exs)
                exs = exs * reps + exs[: target - len(exs) * reps]
            elif len(exs) > target:
                # downsample the dominant buckets by even stride sampling
                step = len(exs) / target
                exs = [exs[int(i * step)] for i in range(target)]
            buckets[b] = exs
        counts[b] = len(exs)
        with open(os.path.join(out_dir, f"train.{b}.jsonl"), "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")

    if balance and buckets:
        total = 0
        with open(os.path.join(out_dir, "train.balanced.jsonl"), "w") as f:
            for b, exs in buckets.items():
                for ex in exs:
                    f.write(json.dumps(ex) + "\n")
                    total += 1
        counts["balanced"] = total
    return counts
