"""Atomic, content-addressed provenance manifests for future finetune runs."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.audit import _load_jsonl, audit_leakage
from src.locking import atomic_write_json

_EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
_ENV_PREFIXES = ("CUDA", "HIP", "HSA", "ROCM", "HF_", "TRANSFORMERS_", "SLURM_")
_ENV_NAMES = {"PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "HOSTNAME"}
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "CREDENTIAL")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_files(path: Path) -> list[Path]:
    return sorted((item for item in path.rglob("*")
                   if item.is_file() and not (_EXCLUDED_PARTS & set(item.parts))),
                  key=lambda item: item.relative_to(path).as_posix())


def hash_path(value: str | Path) -> dict:
    """Return deterministic metadata and SHA-256 for one file or directory tree."""
    path = Path(value)
    entry = {"path": str(path.resolve()) if path.exists() else str(path),
             "kind": "missing", "sha256": None, "size_bytes": 0, "files": 0}
    if path.is_file():
        entry.update(kind="file", sha256=_file_sha256(path),
                     size_bytes=path.stat().st_size, files=1)
        return entry
    if path.is_dir():
        digest = hashlib.sha256()
        size = 0
        files = _tree_files(path)
        for item in files:
            relative = item.relative_to(path).as_posix()
            file_hash = _file_sha256(item)
            size += item.stat().st_size
            digest.update(relative.encode() + b"\0" + file_hash.encode() + b"\n")
        entry.update(kind="directory", sha256=digest.hexdigest(),
                     size_bytes=size, files=len(files))
        return entry
    # A remote model identifier is still recorded reproducibly as an identifier,
    # but is clearly distinguished from verified local bytes.
    text = str(value)
    entry.update(kind="identifier", sha256=hashlib.sha256(text.encode()).hexdigest())
    return entry


def _entries(paths: Iterable[str | Path]) -> list[dict]:
    return [hash_path(path) for path in paths]


def _git(repo_root: Path) -> dict:
    def run(*args: str) -> tuple[int, str]:
        result = subprocess.run(["git", *args], cwd=repo_root, text=True,
                                capture_output=True, check=False)
        return result.returncode, result.stdout.strip() or result.stderr.strip()

    rev_rc, revision = run("rev-parse", "HEAD")
    status_rc, status = run("status", "--porcelain=v1", "--untracked-files=all")
    branch_rc, branch = run("branch", "--show-current")
    return {
        "revision": revision if rev_rc == 0 else None,
        "branch": branch if branch_rc == 0 else None,
        "status": status if status_rc == 0 else None,
        "dirty": bool(status) if status_rc == 0 else None,
        "error": None if rev_rc == status_rc == 0 else revision,
    }


def _environment(environ: Mapping[str, str] | None) -> dict:
    source = dict(os.environ if environ is None else environ)
    result: dict[str, str] = {}
    for key, value in sorted(source.items()):
        if key not in _ENV_NAMES and not key.startswith(_ENV_PREFIXES):
            continue
        result[key] = "<redacted>" if any(mark in key.upper() for mark in _SECRET_MARKERS) else value
    result["python"] = sys.version
    result["platform"] = platform.platform()
    return result


def _trainer(paths: Sequence[str | Path], log_paths: Sequence[str | Path]) -> dict:
    states = []
    logs: list = []
    for value in paths:
        path = Path(value)
        entry = hash_path(path)
        data = None
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
        entry["data"] = data
        states.append(entry)
        if isinstance(data, dict) and isinstance(data.get("log_history"), list):
            logs.extend(data["log_history"])
    log_files = []
    for value in log_paths:
        path = Path(value)
        entry = hash_path(path)
        if path.is_file():
            try:
                entry["tail"] = path.read_text(errors="replace").splitlines()[-200:]
            except OSError:
                entry["tail"] = None
        log_files.append(entry)
    return {"states": states, "log_history": logs, "log_files": log_files}


def _contamination(train_path: str | Path | None,
                   bench_path: str | Path | None) -> dict:
    if train_path is None or bench_path is None:
        return {"status": "not_checked", "reason": "both train and benchmark paths are required"}
    try:
        return audit_leakage(_load_jsonl(str(train_path)), _load_jsonl(str(bench_path)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def build_manifest(*, repo_root: str | Path,
                   datasets: Sequence[str | Path] = (),
                   models: Sequence[str | Path] = (),
                   artifacts: Sequence[str | Path] = (),
                   config_paths: Sequence[str | Path] = (),
                   trainer_state_paths: Sequence[str | Path] = (),
                   trainer_log_paths: Sequence[str | Path] = (),
                   train_path: str | Path | None = None,
                   bench_path: str | Path | None = None,
                   argv: Sequence[str] | None = None,
                   environ: Mapping[str, str] | None = None) -> dict:
    root = Path(repo_root).resolve()
    configs = list(config_paths) or ([root / "config.yaml"] if (root / "config.yaml").exists() else [])
    code = [root / "src"] if (root / "src").exists() else []
    skills = [root / "skills"] if (root / "skills").exists() else []
    docs = sorted([p for p in root.glob("*.md") if p.is_file()] +
                  [p for p in (root / "eval").glob("*.md") if p.is_file()])
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "datasets": _entries(datasets), "models": _entries(models),
            "artifacts": _entries(artifacts), "configs": _entries(configs),
            "code": _entries(code), "skills": _entries(skills), "docs": _entries(docs),
        },
        "git": _git(root),
        "command": {"argv": list(argv if argv is not None else sys.argv),
                    "cwd": os.getcwd(), "executable": sys.executable},
        "environment": _environment(environ),
        "trainer": _trainer(trainer_state_paths, trainer_log_paths),
        "evaluation": {"contamination": _contamination(train_path, bench_path)},
    }


def write_manifest(path: str | Path, manifest: dict) -> Path:
    """Durably replace a manifest; readers see either old or complete new JSON."""
    target = Path(path)
    atomic_write_json(target, manifest)
    return target
