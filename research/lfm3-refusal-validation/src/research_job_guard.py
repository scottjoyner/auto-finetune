"""Fail-closed, read-only per-job capacity gate for LFM3 model evaluations.

Rejects unsafe root and destination free-space conditions before any new
journal or model output. Recheck immediately before each model/generation.
This is not a filesystem reservation or an enforcement tool for other jobs.
"""
from pathlib import Path
import os

from src.research_storage_preflight import preflight

MINIMUM_FREE_GIB = 20
RESERVE_GIB = 8
PLANNED_WRITE_GIB = 5
LOCAL_WORK_ROOT = Path("/home/scott/git")


class ResearchWriteBlocked(RuntimeError):
    """Research model writes are not safe at current free-space levels."""


def existing_directory(path):
    """Find actual filesystem for a possibly not-yet-created output path."""
    path = Path(path).expanduser().absolute()
    while not path.exists():
        parent=path.parent
        if parent == path:
            raise ValueError("output has no existing parent")
        path=parent
    if path.is_symlink() or not path.is_dir():
        raise ValueError("output path resolves to unsafe non-directory endpoint")
    return path
def require_research_write_capacity(output_path, local_root=LOCAL_WORK_ROOT,
                                    planned_write_gib=PLANNED_WRITE_GIB,
                                    reserve_gib=RESERVE_GIB,
                                    minimum_free_gib=MINIMUM_FREE_GIB):
    """A writable NAS destination never excuses an exhausted local work root.

    Returns only after the root and destination (when different devices)
    are both above the minimum. Does not modify the filesystem.
    """
    local=existing_directory(local_root)
    output=existing_directory(output_path)
    if any(v < 0 for v in (planned_write_gib,reserve_gib,minimum_free_gib)):
        raise ValueError("capacity limits must be nonnegative")
    roots=[("local",local)]
    if os.stat(local).st_dev != os.stat(output).st_dev:
        roots.append(("output",output))
    checks={}
    for name,path in roots:
        status=preflight(path,planned_write_gib,reserve_gib,
                         minimum_free_gib)
        checks[name]=status
        if not status["large_write_allowed"]:
            raise ResearchWriteBlocked(
                f"LFM3 research write BLOCKED on {name} filesystem: "
                f"{status['available_gib']:.3f} GiB free, "
                f"{status['required_available_gib']:.3f} GiB required. "
                "Recover and verify storage capacity; no model run started."
            )
    return {
        "model_write_allowed":True,
        "checks":checks,
        "space_reserved":False,
        "other_processes_blocked":False,
    }
