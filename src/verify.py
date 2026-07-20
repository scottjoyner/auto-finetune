"""Verify mined benchmark tasks against their source sessions (CPU-only, safe).

The ``analyze`` step mines executable benchmark tasks into
``<data>/analysis/auto-tasks.jsonl``. Each task records the *instruction* that
was given plus a set of checks (e.g. ``file_contains``) derived from the
successful source session's tool calls.

``verify`` replays those recorded file writes/edits from the source session
into an isolated temporary workspace and runs the checks. It deliberately
executes NO shell / code / web tool calls — only the recorded file
materialization — so it is safe to run anywhere, including while training
occupies the GPU.

This validates two things end-to-end:

  * the task <-> source-session linkage is intact, and
  * the mined check is actually satisfiable by the recorded solution.

The pass-rate is a trackable quality metric for the mining pipeline.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.clean import _dedup_by_session
from src.format_dataset import iter_cleaned_records


def _parse_task_id(task_id: str) -> tuple[str, str]:
    """``auto-<source>-<session_id>`` -> (source, session_id)."""
    if not task_id.startswith("auto-"):
        return "", task_id
    rest = task_id[len("auto-"):]
    source, _, sid = rest.partition("-")
    return source, sid


def load_session_map(cleaned_dir: str) -> dict[str, dict]:
    return _dedup_by_session(iter_cleaned_records(cleaned_dir))


def _patch_added(patch: str) -> tuple[str | None, str | None]:
    """Extract (basename, added-lines) from a unified diff string."""
    path = None
    added: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            path = p
        elif line.startswith("+"):
            added.append(line[1:])
    if not path or not added:
        return None, None
    return os.path.basename(path), "\n".join(added)


def _replay_file_ops(rec: dict, workspace: str) -> list[str]:
    """Materialize the recorded file writes/edits into ``workspace``.

    Handles the real tool shapes seen in the corpus:
      * ``write``/``write_file``/``create_file``/``update_file``
        (``filePath``/``path`` + ``content``)
      * ``edit``/``str_replace`` (``filePath``/``path`` + ``new_string``
        or ``newString``)
      * ``patch`` (unified diff -> added lines)

    Execution tools (bash/python/web/...) and bash redirections are
    intentionally skipped — this is a static replay, not execution — so the
    harness is safe to run anywhere, including while training occupies the GPU.
    """
    written: list[str] = []
    for m in rec.get("messages", []):
        for p in m.get("parts", []):
            if p.get("type") != "tool":
                continue
            name = (p.get("tool") or "").lower()
            inp = p.get("input") or {}
            path = None
            content = None
            if name in ("write", "write_file", "create_file", "update_file"):
                path = inp.get("filePath") or inp.get("path") or inp.get("file")
                content = inp.get("content")
            elif name in ("edit", "str_replace", "str_replace_editor"):
                path = inp.get("filePath") or inp.get("path") or inp.get("file")
                content = inp.get("new_string") or inp.get("newString")
            elif name == "patch":
                pp, pc = _patch_added(inp.get("patch") or "")
                if pp and pc:
                    dest = os.path.join(workspace, pp)
                    with open(dest, "w") as f:
                        f.write(pc)
                    written.append(pp)
                continue
            else:
                continue
            if not isinstance(path, str) or not isinstance(content, str):
                continue
            dest = os.path.join(workspace, os.path.basename(path))
            with open(dest, "w") as f:
                f.write(content)
            written.append(os.path.basename(path))
    return written


def _run_check(check: dict, workspace: str) -> tuple[bool, str]:
    kind = check.get("kind")
    if kind == "file_contains":
        path = os.path.basename(check.get("path") or "")
        expect = check.get("expect") or ""
        dest = os.path.join(workspace, path)
        if not os.path.exists(dest):
            return False, f"file not found: {path}"
        text = Path(dest).read_text(errors="replace")
        if expect in text:
            return True, "ok"
        return False, f"snippet not in {path}"
    return False, f"unsupported check kind: {kind}"


def verify_task(task: dict, sessions: dict[str, dict]) -> dict:
    task_id = task.get("task_id", "")
    source, sid = _parse_task_id(task_id)
    rec = sessions.get(sid)
    if rec is None:
        return {"task_id": task_id, "ok": False,
                "reason": "source session not found", "replayed": [], "checks": []}
    with tempfile.TemporaryDirectory() as ws:
        replayed = _replay_file_ops(rec, ws)
        checks = []
        for c in task.get("checks", []):
            ok, detail = _run_check(c, ws)
            checks.append({"kind": c.get("kind"), "ok": ok, "detail": detail})
    ok = bool(checks) and all(c["ok"] for c in checks)
    reason = ("ok" if ok
              else "; ".join(c["detail"] for c in checks if not c["ok"]) or "no checks")
    return {"task_id": task_id, "source": source, "bucket": task.get("bucket"),
            "difficulty": task.get("difficulty"), "ok": ok, "reason": reason,
            "replayed": replayed, "checks": checks}


def verify_all(tasks_path: str, cleaned_dir: str) -> list[dict]:
    sessions = load_session_map(cleaned_dir)
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    return [verify_task(t, sessions) for t in tasks]


def summarize(results: list[dict]) -> dict:
    n = len(results)
    found = sum(1 for r in results if r["reason"] != "source session not found")
    passed = sum(1 for r in results if r["ok"])
    unsupported = sum(1 for r in results
                     for c in r["checks"] if c["detail"].startswith("unsupported"))
    return {"n_tasks": n, "sessions_found": found, "checks_passed": passed,
            "unsupported_checks": unsupported,
            "pass_rate": round(passed / n, 3) if n else 0.0}
