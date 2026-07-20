"""CPU-only corpus analysis over already-cleaned sessions.

Produces, without touching the GPU or the live training datasets:

  * task-type **buckets**      (shell / file-edit / multi-file-refactor /
                                code-search / debug / web-research / data-analysis /
                                docs / reasoning / mixed) via a deterministic
                                heuristic classifier (tool histogram + file types +
                                intent keywords).
  * **difficulty** tiers       (easy / medium / hard) from turn/tool signals.
  * **quality flags**          drop too-short / empty sessions.
  * **auto benchmark tasks**   mined from successful file-edit sessions.
  * **failure / negative set** sessions containing unresolved tool errors.
  * **corpus stats**           bucket/source/difficulty counts, tool & file-type
                                frequency, dedup rate, and opencode<->hermes overlap.

Outputs go to a staging dir (default: <data>/analysis) so `datasets/` — which a
live training run memory-maps — is never disturbed. See HARVEST.md.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.clean import _conv_hash, _dedup_by_session

# ── tool taxonomy ──────────────────────────────────────────────────────────────
EDIT_TOOLS = {"write", "edit", "patch", "str_replace", "create_file", "update_file",
              "str_replace_editor", "write_file"}
SHELL_TOOLS = {"bash", "shell", "terminal", "sh", "zsh", "cmd", "powershell", "execute"}
SEARCH_TOOLS = {"grep", "rg", "glob", "find", "search", "ls"}
READ_TOOLS = {"read", "cat", "view", "open", "open_file"}
WEB_TOOLS = {"web", "fetch", "browser", "browse", "http_get", "scrape", "websearch"}
CODE_TOOLS = {"python", "jupyter", "ipython", "notebook", "repl", "execute_python"}

# Deliberately specific — bare "error"/"failed" alone over-fire on benign
# output (e.g. "build failed" is common and not a session failure).
_ERROR_MARKERS = ("traceback", "error:", " exception", "command not found",
                  "permission denied", "no such file or directory",
                  "syntaxerror", "exit code", "errno", "module not found",
                  " is not recognized", "denied:")


def _is_error(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ERROR_MARKERS)


def _tool_error(grp: str, out: Any) -> str | None:
    """Return an error snippet if ``out`` is a tool failure, else ``None``.

    Structure-aware. Shell/exec tools emit JSON strings carrying an
    ``exit_code`` field; ``exit_code == 0`` is a success even when the
    text mentions "error" (e.g. ``curl: (22) ... error: 422``), which
    is the dominant false-positive source in the corpus. Non-zero
    ``exit_code`` is a confirmed failure. Plain (non-JSON) outputs
    fall back to substring marker scanning, applied only to executable
    tools -- read tools return file contents that routinely contain the
    word "error".
    """
    if grp == "read":
        return None
    if isinstance(out, dict):
        if out.get("error"):
            return str(out.get("error"))
        if out.get("success") is False:
            return str(out.get("content") or out.get("error") or "failed")
        ec = out.get("exit_code")
        if ec == 0 or ec == "0":
            return None
        if ec not in (None, ""):
            return str(out.get("content") or f"exit_code {ec}")
        return _is_error_str(str(out.get("content", "")))
    if isinstance(out, str):
        s = out.strip()
        if s.startswith("{") and "exit_code" in s:
            try:
                d = json.loads(s)
            except Exception:
                d = None
            if isinstance(d, dict):
                ec = d.get("exit_code")
                if ec == 0 or ec == "0":
                    return None
                if d.get("error"):
                    return str(d.get("error"))
                if ec not in (None, ""):
                    return s
                s = str(d.get("output", s))
        return _is_error_str(s)
    return None


def _is_error_str(text: str) -> str | None:
    return text if _is_error(text) else None


def _group(name: str) -> str:
    n = name.lower()
    if n in SHELL_TOOLS:
        return "shell"
    if n in EDIT_TOOLS:
        return "edit"
    if n in SEARCH_TOOLS:
        return "search"
    if n in READ_TOOLS:
        return "read"
    if n in WEB_TOOLS:
        return "web"
    if n in CODE_TOOLS:
        return "code"
    return "other"


_INTENT_KEYWORDS = {
    "debug": ("fix", "bug", "debug", "traceback", "broken", "failing", "why does",
              "not working", "error"),
    "refactor": ("refactor", "restructure", "reorganize", "clean up the code",
                 "rename"),
    "implement": ("implement", "create", "build", "add a", "add an", "scaffold",
                  "write a", "develop"),
    "docs": ("readme", "documentation", "docs", "docstring", "comment the",
             "write docs"),
    "research": ("research", "find out", "look up", "what is", "how do", "explain",
                 "why is", "compare"),
    "data": ("csv", "dataframe", "pandas", "analyze", "plot", "chart",
             "statistics", "dataset", "xlsx", "parquet"),
    "web": ("http", "url", "website", "fetch", "browse", "search the web"),
}


def _intent(text: str) -> list[str]:
    t = text.lower()
    out = []
    for tag, kws in _INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            out.append(tag)
    return out


def _exts_from_input(inp: Any) -> set[str]:
    exts: set[str] = set()
    if not isinstance(inp, dict):
        return exts
    for k in ("filePath", "path", "filename", "filenames", "file"):
        v = inp.get(k)
        if isinstance(v, str) and v:
            exts.add(os.path.splitext(v)[1].lower())
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x:
                    exts.add(os.path.splitext(os.path.basename(x))[1].lower())
    return exts


def _paths_from_input(inp: Any) -> set[str]:
    paths: set[str] = set()
    if not isinstance(inp, dict):
        return paths
    for k in ("filePath", "path", "filename", "filenames", "file"):
        v = inp.get(k)
        if isinstance(v, str) and v:
            paths.add(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x:
                    paths.add(x)
    return paths


def extract_features(rec: dict) -> dict:
    """Pull classification features from one normalized session record."""
    tool_hist: Counter = Counter()
    exts: set[str] = set()
    paths: set[str] = set()
    n_tool = 0
    n_turn = 0
    n_user_text = 0
    total_chars = 0
    errors: list[str] = []
    first_user_text = ""

    for m in rec.get("messages", []):
        n_turn += 1
        role = m.get("role")
        for p in m.get("parts", []):
            t = p.get("type")
            if t == "tool":
                n_tool += 1
                name = (p.get("tool") or "").lower()
                tool_hist[name] += 1
                exts |= _exts_from_input(p.get("input"))
                paths |= _paths_from_input(p.get("input"))
                out = p.get("output")
                # Structure-aware error detection (see ``_tool_error``):
                # executable tools only, with `exit_code`/`success`/`error`
                # honored when present.
                grp = _group(name)
                err = _tool_error(grp, out)
                if err:
                    errors.append(err)
            elif t == "text" and isinstance(p.get("text"), str):
                txt = p["text"]
                total_chars += len(txt)
                if role == "user":
                    n_user_text += 1
                    if not first_user_text:
                        first_user_text = txt
                else:
                    n_user_text += 0
            elif t == "patch":
                exts |= _exts_from_input(p)
                paths |= _paths_from_input(p)

    groups: Counter = Counter()
    for name, c in tool_hist.items():
        groups[_group(name)] += c

    return {
        "tool_hist": dict(tool_hist),
        "groups": dict(groups),
        "exts": sorted(exts),
        "n_turns": n_turn,
        "n_tool": n_tool,
        "tool_diversity": len(tool_hist),
        "n_user_text": n_user_text,
        "total_chars": total_chars,
        "intent": _intent(first_user_text),
        "has_error": bool(errors),
        "error_snippet": (errors[0][:200] if errors else ""),
        "distinct_files": len(paths),
        "first_user_text": first_user_text[:300],
    }


def classify_bucket(f: dict) -> str:
    """Map features to a task-type bucket (priority cascade)."""
    g = f["groups"]
    intent = set(f["intent"])

    if g.get("web", 0) > 0:
        return "web-research"
    # debug intent wins even when the session also edits files
    if {"debug"} & intent:
        return "debug"
    if g.get("edit", 0) > 0:
        return "multi-file-refactor" if f["distinct_files"] >= 3 else "file-edit"
    if g.get("search", 0) > 0 and g.get("edit", 0) == 0 and g.get("shell", 0) == 0:
        return "code-search"
    if g.get("shell", 0) >= 3 and g.get("edit", 0) == 0:
        return "shell"
    if ({"data"} & intent) or any(e in {".csv", ".tsv", ".xlsx", ".parquet", ".json"}
                                  for e in f["exts"]):
        return "data-analysis"
    if ({"docs"} & intent) or any(e == ".md" for e in f["exts"]):
        return "docs"
    if f["n_tool"] <= 1:
        return "reasoning"
    # only fall back to debug for errored sessions no other signal claimed
    if f["has_error"]:
        return "debug"
    return "mixed"


def classify_difficulty(f: dict) -> str:
    turns = f["n_turns"]
    tc = f["n_tool"]
    td = f["tool_diversity"]
    if turns <= 6 and tc <= 4:
        return "easy"
    if turns >= 25 or tc >= 12 or td >= 6:
        return "hard"
    return "medium"


def quality_flag(f: dict, rec: dict) -> tuple[bool, str]:
    if f["n_turns"] < 3:
        return False, "too_short"
    if f["n_tool"] == 0 and f["n_user_text"] == 0:
        return False, "no_content"
    return True, "ok"


def _file_target(inp: dict) -> tuple[str, str] | None:
    """Extract (basename, content-snippet) a tool call produces.

    Handles the real tool shapes seen in the corpus: ``write``/``write_file``
    (``filePath``/``path`` + ``content``), ``edit``/``str_replace``
    (``filePath``/``path`` + ``new_string``/``newString``) and unified
    ``patch`` parts (``+++ b/<path>`` + added lines).
    """
    if not isinstance(inp, dict):
        return None
    path = inp.get("filePath") or inp.get("path") or inp.get("file") or inp.get("filename")
    content = (inp.get("content") or inp.get("new_string") or inp.get("newString") or "")
    if isinstance(content, str) and isinstance(path, str) and content.strip():
        return os.path.basename(path), content.strip()[:60]
    # patch: pull the file path and the added (+) lines
    pc = inp.get("patch")
    if isinstance(pc, str):
        p = None
        added = []
        for line in pc.splitlines():
            if line.startswith("+++ "):
                p = line[4:].strip()
                if p.startswith("b/"):
                    p = p[2:]
            elif line.startswith("+"):
                added.append(line[1:])
        if p and added:
            return os.path.basename(p), "\n".join(added).strip()[:60]
    return None


def derive_task(rec: dict, meta: dict) -> dict | None:
    """Turn a successful file-edit session into a seed benchmark task."""
    if not meta["keep"]:
        return None
    if meta["bucket"] not in ("file-edit", "multi-file-refactor", "docs", "shell", "debug"):
        return None
    instr = (meta["features"].get("first_user_text") or "").strip()
    if len(instr) < 20:
        return None
    target = None
    for m in rec.get("messages", []):
        for p in m.get("parts", []):
            if p.get("type") == "tool" and (p.get("tool") or "").lower() in EDIT_TOOLS:
                tgt = _file_target(p.get("input") or {})
                if tgt:
                    target = tgt
                    break
        if target:
            break
    if not target:
        return None
    path, snippet = target
    return {
        "task_id": f"auto-{meta['source']}-{meta['session_id']}",
        "kind": "exec",
        "instruction": instr[:500],
        "checks": [{"kind": "file_contains", "path": path, "expect": snippet}],
        "source": meta["source"],
        "bucket": meta["bucket"],
        "difficulty": meta["difficulty"],
        "auto": True,
    }


def benchmark_session_ids(tasks_path: str | Path) -> set[str]:
    """Recover the source session ids behind mined auto-tasks.

    An auto-task ``task_id`` is ``auto-{source}-{session_id}`` (see
    ``derive_task``). This inverts it so those sessions can be held
    out of the training mix for a true benchmark -- otherwise the
    49-task eval would overlap the SFT corpus.

    Returns an empty set (not an error) when ``tasks_path`` is absent.
    """
    ids: set[str] = set()
    p = Path(tasks_path)
    if not p.exists():
        return ids
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        tid = t.get("task_id", "")
        if not tid.startswith("auto-"):
            continue
        prefix = f"auto-{t.get('source', '')}-"
        if tid.startswith(prefix):
            ids.add(tid[len(prefix):])
    return ids


def compute_stats(metas: list[dict]) -> dict:
    by_source: Counter = Counter()
    by_bucket: Counter = Counter()
    by_diff: Counter = Counter()
    tool_counter: Counter = Counter()
    ext_counter: Counter = Counter()
    turns: list[int] = []
    has_error = 0
    all_hashes: set[str] = set()
    per_source_hashes: dict[str, set[str]] = {}

    for m in metas:
        by_source[m["source"]] += 1
        by_bucket[m["bucket"]] += 1
        by_diff[m["difficulty"]] += 1
        for t, c in m["features"]["tool_hist"].items():
            tool_counter[t] += c
        for e in m["features"]["exts"]:
            ext_counter[e] += 1
        turns.append(m["features"]["n_turns"])
        if m["features"]["has_error"]:
            has_error += 1
        h = m["hash"]
        all_hashes.add(h)
        per_source_hashes.setdefault(m["source"], set()).add(h)

    overlap = len(per_source_hashes.get("opencode", set()) &
                 per_source_hashes.get("hermes", set()))
    avg = (sum(turns) / len(turns)) if turns else 0.0
    total = len(metas)
    return {
        "total": total,
        "by_source": dict(by_source.most_common()),
        "by_bucket": dict(by_bucket.most_common()),
        "by_difficulty": dict(by_diff),
        "top_tools": tool_counter.most_common(15),
        "top_exts": ext_counter.most_common(15),
        "avg_turns": round(avg, 1),
        "error_rate": round(has_error / total, 3) if total else 0.0,
        "unique_rate": round(len(all_hashes) / total, 3) if total else 0.0,
        "cross_source_overlap": overlap,
    }


def load_sessions(cleaned_dir: str) -> list[dict]:
    recs: list[dict] = []
    for path in sorted(Path(cleaned_dir).rglob("*.json")):
        try:
            recs.append(json.loads(path.read_text()))
        except Exception:
            continue
    return recs


def analyze_all(cleaned_dir: str, out_dir: str | None = None) -> dict:
    """Run the full analysis and write the staging artifacts.

    Writes (into ``out_dir``):
      buckets.json        session_id -> {source,bucket,difficulty,keep,quality_reason}
      corpus.json         aggregate stats
      auto-tasks.jsonl    mined benchmark tasks (seed quality)
      failures.jsonl      sessions with unresolved tool errors

    The corpus is deduplicated by ``session_id`` (see ``src.clean._dedup_by_session``)
    so cross-source / snapshot duplicates count once.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(str(cleaned_dir)), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    recs = list(_dedup_by_session(load_sessions(cleaned_dir)).values())
    metas: list[dict] = []
    paired: list[tuple[dict, dict]] = []
    for rec in recs:
        f = extract_features(rec)
        bucket = classify_bucket(f)
        diff = classify_difficulty(f)
        keep, reason = quality_flag(f, rec)
        meta = {
            "session_id": rec.get("session_id") or "",
            "source": rec.get("source") or "",
            "bucket": bucket,
            "difficulty": diff,
            "keep": keep,
            "quality_reason": reason,
            "features": f,
            "hash": _conv_hash(rec),
        }
        metas.append(meta)
        paired.append((rec, meta))

    tasks = []
    for rec, meta in paired:
        if len(tasks) >= 80:
            break
        t = derive_task(rec, meta)
        if t:
            tasks.append(t)

    failures = [
        {"session_id": m["session_id"], "source": m["source"], "bucket": m["bucket"],
         "difficulty": m["difficulty"], "error": m["features"]["error_snippet"]}
        for m in metas if m["features"]["has_error"]
    ]

    stats = compute_stats(metas)

    buckets_out = {
        m["session_id"]: {k: m[k] for k in ("source", "bucket", "difficulty", "keep", "quality_reason")}
        for m in metas
    }
    (Path(out_dir) / "buckets.json").write_text(json.dumps(buckets_out, indent=1))
    (Path(out_dir) / "corpus.json").write_text(json.dumps(stats, indent=2))
    with (Path(out_dir) / "auto-tasks.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    with (Path(out_dir) / "failures.jsonl").open("w") as f:
        for fl in failures:
            f.write(json.dumps(fl) + "\n")

    return {
        "n_sessions": len(metas),
        "n_tasks": len(tasks),
        "n_failures": len(failures),
        "out_dir": out_dir,
        "stats": stats,
    }
