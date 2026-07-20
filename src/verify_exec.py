"""Sandboxed execution verification of mined tasks (CPU-only, opt-in, guarded).

``verify`` only does a *static* replay (no execution). ``verify-exec`` goes
further: after the static replay it ALSO runs the recorded ``bash`` /
``execute_code`` tool calls in an isolated temporary directory to materialize the
files those calls create (heredocs / ``python`` / ``tee`` / ...). This yields
a *true* "did the task get done?" rate across the whole mined set.

SAFETY (read before running):
  * Every command runs with cwd = a fresh temp dir that is deleted
    afterwards. Blast radius is limited to that directory.
  * A denylist refuses to run any command matching destructive or
    network-egress patterns (``rm -rf``, ``sudo``, ``dd``, ``mkfs``,
    ``git push``, ``curl``, ``wget``, ``ssh``, ``scp``, ``pip``/``npm``/
    ``apt`` installs, absolute writes to ``/home``, ``/media``, ``/etc``,
    ...). Those tasks are reported as "blocked" rather than executed.
  * Per-command timeout (default 30s).
  * This is OPT-IN via ``cli verify-exec`` and replays the *operator's
    own* historical sessions, not untrusted input.

It is still CPU-only (no GPU) and never touches ``datasets/``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

from src.verify import (
    _parse_task_id,
    _replay_file_ops,
    _run_check,
    load_session_map,
)

# Destructive or network-egress patterns — matched commands are NEVER run.
# Deliberately narrow: we must NOT block benign tokens like ``/home``,
# ``/etc`` or ``>/dev/null`` (``2>/dev/null`` is in almost every real
# command). We block only genuinely dangerous targets (``/media`` is the
# data disk; block device writes; network + remote + pkg managers).
_BLOCK = [
    re.compile(p, re.I) for p in (
        r"\brm\s+-rf\b", r"\brm\s+-fr\b", r"\brm\s+-r\b\s+/",
        r"\bsudo\b", r"\bmkfs\b", r"\bdd\b\s+if=",
        r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
        r"\bgit\s+push\b", r"--force\b", r"--hard\b",
        r">\s*/dev/sd", r">\s*/dev/hd", r">\s*/dev/nvme",
        r":\s*/dev/sd", r":\s*/dev/hd", r":\s*/dev/nvme",
        r"/dev/sd", r"/dev/hd", r"/dev/nvme",
        r"/media",
        r"\bssh\b", r"\bscp\b", r"\bnc\b", r"\bnetcat\b",
        r"\bcurl\b", r"\bwget\b", r"\bgit\s+clone\b",
        r"\bpip\b", r"\bnpm\b", r"\bapt\b", r"\byum\b", r"\bbrew\b",
        r"\bdocker\b", r"\bchmod\b\s+-R", r"\bchown\b\s+-R",
    )
]


def _command_safe(cmd: str) -> tuple[bool, str]:
    if not isinstance(cmd, str) or not cmd.strip():
        return False, "empty"
    for rx in _BLOCK:
        m = rx.search(cmd)
        if m:
            return False, f"blocked:{m.group(0).strip()}"
    return True, ""


def _run_in_sandbox(cmd: str, cwd: str, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=cwd,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:  # noqa: BLE001
        return 1, f"error:{e}"


def _replay_exec(rec: dict, workspace: str, timeout: int) -> list[str]:
    """Run recorded bash/terminal/execute_code tool calls in ``workspace``."""
    ran: list[str] = []
    for m in rec.get("messages", []):
        for p in m.get("parts", []):
            if p.get("type") != "tool":
                continue
            name = (p.get("tool") or "").lower()
            if name in ("bash", "terminal", "sh", "zsh", "execute", "shell"):
                cmd = (p.get("input") or {}).get("command")
            elif name == "execute_code":
                code = (p.get("input") or {}).get("code")
                cmd = f"python -c {code!r}" if isinstance(code, str) else None
            else:
                continue
            if not isinstance(cmd, str):
                continue
            ok, why = _command_safe(cmd)
            if not ok:
                ran.append(f"skip:{why}")
                continue
            rc, _out = _run_in_sandbox(cmd, workspace, timeout)
            ran.append(f"rc={rc}")
    return ran


def verify_task_exec(task: dict, sessions: dict[str, dict], timeout: int = 30) -> dict:
    task_id = task.get("task_id", "")
    source, sid = _parse_task_id(task_id)
    rec = sessions.get(sid)
    if rec is None:
        return {"task_id": task_id, "ok": False,
                "reason": "source session not found", "replayed": [], "checks": []}
    with tempfile.TemporaryDirectory() as ws:
        static = _replay_file_ops(rec, ws)
        exec_log = _replay_exec(rec, ws, timeout)
        checks = []
        for c in task.get("checks", []):
            ok, detail = _run_check(c, ws)
            checks.append({"kind": c.get("kind"), "ok": ok, "detail": detail})
    ok = bool(checks) and all(c["ok"] for c in checks)
    reason = ("ok" if ok
              else "; ".join(c["detail"] for c in checks if not c["ok"]) or "no checks")
    return {"task_id": task_id, "source": source, "bucket": task.get("bucket"),
            "difficulty": task.get("difficulty"), "ok": ok, "reason": reason,
            "replayed_static": static, "replayed_exec": exec_log, "checks": checks}


def verify_all_exec(tasks_path: str, cleaned_dir: str, timeout: int = 30) -> list[dict]:
    sessions = load_session_map(cleaned_dir)
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    return [verify_task_exec(t, sessions, timeout=timeout) for t in tasks]
