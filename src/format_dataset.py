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
from src.locking import atomic_write_json


def _atomic_text(path: "str | Path", text: str) -> None:
    """Replace a text file atomically (tmp file in the same dir, then rename)."""
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp",
                                    dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


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


def _write_dataset(out_path: str, rows: list, sids: list[str]) -> None:
    """Write a dataset plus its provenance sidecar, row-aligned.

    Training rows carry only ``messages``; the sidecar is what makes it possible
    to confirm later that no held-out benchmark session is inside them. A
    session may contribute several rows (one per window), so ``sids`` repeats.
    """
    with open(out_path, "w") as f:
        for ex in rows:
            f.write(json.dumps(ex) + "\n")
    base = out_path[:-len(".jsonl")] if out_path.endswith(".jsonl") else out_path
    with open(base + ".provenance.jsonl", "w") as pf:
        for i, sid in enumerate(sids):
            pf.write(json.dumps({"row": i, "session_id": sid}) + "\n")


def main(cfg: Config, source: str | None = None, label: str | None = None,
         exclude: set[str] | None = None) -> int:
    """Emit training datasets from the cleaned corpus.

    ``exclude`` is a set of session ids held out of every output (used for
    benchmark sessions). This path previously had no holdout at all, so
    candidate corpora contained the eval set regardless of what the analyzer
    resolved.
    """
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")
    os.makedirs(dataset_dir, exist_ok=True)

    template = cfg.get("format", "template", default="chatml")
    system = cfg.get("format", "system_prompt", default="") or ""
    max_turns = cfg.get("format", "max_turns_per_example", default=0) or 0
    max_chars = cfg.get("format", "max_chars_per_example", default=24000) or 0
    # Session kinds to drop entirely (e.g. automated cron runs: highly
    # repetitive, low information density, and they dominate session counts).
    exclude_sources = set(cfg.get("format", "exclude_sources", default=[]) or [])
    kept_counts: dict[str, int] = {}
    dropped_counts: dict[str, int] = {}
    held_out_totals = [0]

    def _format_one(src_dir: str, out_path: str, filter_source: str | None) -> int:
        examples: list[Any] = []
        sids: list[str] = []
        sources_seen: set[str] = set()
        held_out = 0
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(src_dir, fn)) as f:
                rec = json.load(f)
            src = rec.get("source", "")
            agent = str(rec.get("agent") or "")
            sources_seen.add(src)
            sid = str(rec.get("session_id") or os.path.basename(fn)[:-5])
            if exclude and sid in exclude:
                held_out += 1
                continue
            # Exclusion matches the session's source OR its agent kind
            # (automated Hermes cron runs carry source='hermes' but
            # agent='cron').
            if (src in exclude_sources or agent in exclude_sources) and \
                    (not filter_source or filter_source == src):
                dropped_counts[agent or src] = dropped_counts.get(agent or src, 0) + 1
                continue
            if filter_source and src != filter_source:
                continue
            kept_counts[src] = kept_counts.get(src, 0) + 1
            msgs = rec.get("messages", [])
            windows = _window_messages(msgs, max_turns, max_chars)
            for w in windows:
                if len(w) < 2:
                    continue
                examples.append(_format_window(w, template, system))
                sids.append(sid)
        # A session may contribute one row per window, so sids repeats; no
        # deduping here, as that would collapse legitimate windows.
        _write_dataset(out_path, examples, sids)
        held_out_totals[0] += held_out
        return len(examples)

    def _collect_examples(src_dir: str, max_turns: int, max_chars: int,
                          template: str, system: str) -> tuple[list, list[str]]:
        examples: list[Any] = []
        sids: list[str] = []
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(src_dir, fn)) as f:
                rec = json.load(f)
            if rec.get("source") != "opencode":
                continue
            sid = str(rec.get("session_id") or os.path.basename(fn)[:-5])
            if exclude and sid in exclude:
                held_out_totals[0] += 1
                continue
            msgs = rec.get("messages", [])
            windows = _window_messages(msgs, max_turns, max_chars)
            for w in windows:
                if len(w) < 2:
                    continue
                examples.append(_format_window(w, template, system))
                sids.append(sid)
        return examples, sids

    if label == "opencode-all":
        # Merge every opencode source subdir (ssd, nas5-*, opencode-<project>)
        # into one corpus. Identified by peeking each cleaned subdir's records.
        out_path = os.path.join(dataset_dir, "train.opencode-all.jsonl")
        examples: list[Any] = []
        sids: list[str] = []
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
            got, got_sids = _collect_examples(src_dir, max_turns, max_chars,
                                              template, system)
            examples.extend(got)
            sids.extend(got_sids)
        _write_dataset(out_path, examples, sids)
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
    if exclude:
        print(f"[format] held out {held_out_totals[0]} benchmark session(s) "
              f"(recorded per row in *.provenance.jsonl)")
    if exclude_sources:
        print(f"[format] excluded sources: "
              + ", ".join(f"{k}={v} sessions dropped" for k, v in sorted(dropped_counts.items()))
              + f" | kept: " + ", ".join(f"{k}={v}" for k, v in sorted(kept_counts.items())))
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


def combine(cfg: Config, exclude: set[str] | None = None) -> int:
    """Merge all per-source datasets into one train.combined.jsonl.

    This is the "finetune them all together" equivalent: a single corpus over
    the union of sources, ready for one LoRA run with --label=combined.

    Carries each source row's provenance through the dedupe so the merged corpus
    stays auditable, and drops any held-out benchmark session. Sources lacking a
    sidecar are merged but reported unverifiable rather than assumed clean.
    """
    dataset_dir = cfg.path("dataset_dir")
    out_path = os.path.join(dataset_dir, "train.combined.jsonl")
    seen: set[str] = set()
    out_rows: list[str] = []
    out_sids: list[str] = []
    missing_provenance: list[str] = []
    total = 0
    for label in _COMBINE_LABELS:
        src = os.path.join(dataset_dir, f"train.{label}.jsonl")
        if not os.path.exists(src):
            print(f"[combine] skip missing {src}")
            continue
        sids = read_provenance(src)
        if sids is None:
            missing_provenance.append(label)
            sids = ["" for _ in range(sum(1 for _ in open(src)))]
        n = 0
        for i, line in enumerate(open(src)):
            line = line.strip()
            if not line:
                continue
            # de-dupe identical examples across sources
            if line in seen:
                continue
            sid = sids[i] if i < len(sids) else ""
            if exclude and sid and sid in exclude:
                continue
            seen.add(line)
            out_rows.append(line)
            out_sids.append(sid)
            n += 1
        print(f"[combine] {label}: {n} examples")
        total += n
    _atomic_text(out_path, "".join(r + "\n" for r in out_rows))
    write_provenance(out_path, out_sids)
    print(f"[combine] wrote {total} unique examples -> {out_path}")
    if missing_provenance:
        print(f"[combine] WARNING no provenance sidecar for: "
              f"{', '.join(missing_provenance)} (holdout unverifiable for those)")
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
                balance: bool = False, cap: int | None = None,
                exclude: set[str] | None = None) -> dict:
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
    # Rows carry only `messages`, so once emitted there is no way to tell which
    # session an example came from -- which is why a held-out benchmark session
    # can end up back in the mix unnoticed. Carry the session id alongside each
    # example as a sidecar instead of inside the training row: same provenance,
    # no change to what the trainer reads.
    buckets: dict[str, list] = defaultdict(list)
    excluded = 0
    for sid, rec in unique.items():
        if exclude and sid in exclude:
            excluded += 1
            continue
        b = (bucket_map.get(sid) or {}).get("bucket")
        if not b:
            from src import analyze as _a
            b = _a.classify_bucket(_a.extract_features(rec))
        for w in _window_messages(rec.get("messages", []), max_turns, max_chars):
            if len(w) < 2:
                continue
            buckets[b].append((_format_window(w, template, system), sid))

    target = cap if cap else (max(len(v) for v in buckets.values()) if buckets else 0)
    counts: dict[str, int] = {}
    provenance: dict[str, dict] = {}
    for b, rows in sorted(buckets.items()):
        if balance and target:
            if len(rows) < target:
                # upsample by repetition
                reps = target // len(rows)
                rows = rows * reps + rows[: target - len(rows) * reps]
            elif len(rows) > target:
                # downsample the dominant buckets by even stride sampling
                step = len(rows) / target
                rows = [rows[int(i * step)] for i in range(target)]
            buckets[b] = rows
        counts[b] = len(rows)
        # Balance transforms run on (example, session_id) pairs, so row order
        # stays aligned with the provenance sidecar written below.
        sessions = sorted({sid for _, sid in rows})
        provenance[b] = {"path": os.path.join(out_dir, f"train.{b}.jsonl"),
                         "rows": len(rows), "sessions": sessions}
        with open(os.path.join(out_dir, f"train.{b}.jsonl"), "w") as f, \
             open(os.path.join(out_dir, f"train.{b}.provenance.jsonl"), "w") as pf:
            for i, (ex, sid) in enumerate(rows):
                f.write(json.dumps(ex) + "\n")
                pf.write(json.dumps({"row": i, "session_id": sid}) + "\n")

    if balance and buckets:
        total = 0
        with open(os.path.join(out_dir, "train.balanced.jsonl"), "w") as f:
            for b, rows in buckets.items():
                for ex, _sid in rows:
                    f.write(json.dumps(ex) + "\n")
                    total += 1
        counts["balanced"] = total
    if exclude is not None:
        counts["excluded"] = excluded
    from src.locking import atomic_write_json
    atomic_write_json(os.path.join(out_dir, "strata-manifest.json"),
                      {"counts": counts, "strata": provenance,
                       "excluded_sessions": sorted(exclude) if exclude else []})
    return counts


def read_provenance(dataset_path: str | Path) -> list[str] | None:
    """Return the session id per row for a dataset, or None if unavailable.

    Sidecars are written by :func:`_write_dataset` and :func:`emit_strata`. A
    dataset emitted before provenance existed has none, and callers must treat
    that as unverifiable rather than clean.
    """
    p = Path(dataset_path)
    sidecar = p.with_suffix(".provenance.jsonl")
    if not sidecar.is_file():
        return None
    sids: list[str] = []
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sids.append(str(json.loads(line).get("session_id") or ""))
    return sids


def write_provenance(dataset_path: str | Path, sids: list[str]) -> None:
    """Write the row-aligned provenance sidecar for an already-written dataset.

    Raises on a row-count mismatch: a sidecar that does not match its dataset
    would make every later verification meaningless.
    """
    dataset_path = Path(dataset_path)
    n_rows = sum(1 for line in dataset_path.open() if line.strip())
    if len(sids) != n_rows:
        raise ValueError(
            f"provenance length {len(sids)} does not match "
            f"{dataset_path.name} row count {n_rows}")
    tmp = dataset_path.with_suffix(".provenance.jsonl.tmp")
    with open(tmp, "w") as f:
        for i, sid in enumerate(sids):
            f.write(json.dumps({"row": i, "session_id": sid}) + "\n")
    os.replace(tmp, dataset_path.with_suffix(".provenance.jsonl"))


def verify_dataset(dataset_path: str | Path, held_out: set[str] | None) -> dict:
    """Check one dataset's provenance sidecar for held-out sessions.

    Returns ``{"status", "leaked", "n_rows", "n_sessions"}`` where status is
    ``clean``, ``contaminated``, or ``unverifiable`` when no sidecar exists.
    """
    held_out = held_out or set()
    path = Path(dataset_path)
    sids = read_provenance(path)
    if sids is None:
        return {"status": "unverifiable", "leaked": [], "n_rows": 0, "n_sessions": 0}
    n_rows = sum(1 for _ in path.open())
    if len(sids) != n_rows:
        return {"status": "contaminated", "n_rows": n_rows, "n_sessions": 0,
                "leaked": [{"reason": f"provenance has {len(sids)} rows, "
                                      f"dataset has {n_rows}"}]}
    leaked = [{"row": i, "session_id": sid} for i, sid in enumerate(sids)
              if sid and sid in held_out]
    return {"status": "contaminated" if leaked else "clean",
            "leaked": leaked[:50], "n_rows": n_rows,
            "n_sessions": len({s for s in sids if s})}


def verify_holdout(out_dir: str, held_out: set[str] | None) -> dict:
    """Check that no emitted stratum row came from a held-out session.

    Reads the ``*.provenance.jsonl`` sidecars written by :func:`emit_strata`.
    Returns ``{"status", "leaked", "n_rows", "n_sessions", "checked"}``; ``status``
    is ``"clean"``, ``"contaminated"``, or ``"unverifiable"`` when a sidecar is
    missing (which is the case for strata emitted before this existed).
    """
    held_out = held_out or set()
    root = Path(out_dir)
    leaked: list[dict] = []
    n_rows = 0
    n_sessions = 0
    # `train.provenance.jsonl` (merged) alongside `train.<label>.provenance.jsonl`
    files = sorted(root.glob("train*.provenance.jsonl"))
    if not files:
        return {"status": "unverifiable", "leaked": [], "n_rows": 0,
                "n_sessions": 0, "checked": []}
    for pf in files:
        target = root / pf.name.replace(".provenance.jsonl", ".jsonl")
        if not target.is_file():
            leaked.append({"file": target.name, "reason": "provenance without dataset"})
            continue
        checked = 0
        sessions: set[str] = set()
        rows_in_dataset = sum(1 for _ in target.open())
        for line in pf.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("session_id") or ""
            checked += 1
            n_rows += 1
            sessions.add(sid)
            if sid and sid in held_out:
                leaked.append({"file": target.name, "row": rec.get("row"),
                               "session_id": sid})
        if checked != rows_in_dataset:
            leaked.append({"file": target.name, "reason":
                           f"provenance has {checked} rows, dataset has {rows_in_dataset}"})
        n_sessions += len(sessions)
    return {"status": "contaminated" if leaked else "clean",
            "leaked": leaked[:50], "n_rows": n_rows, "n_sessions": n_sessions,
            "checked": [p.name for p in files]}
