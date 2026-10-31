"""Opt-in Cargo build runner with per-worktree targets on a separate SSD.

Planning is read-only. Executing requires an explicit flag and checks
both root and SSD headroom. Never alters global Cargo configuration,
stops builds, moves existing targets, or changes filesystem reserves.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from src.research_storage_preflight import GIB

DEFAULT_SSD = Path("/media/scott/SSD_4TB")
DEFAULT_CARGO = Path("/home/scott/.cargo/bin/cargo")
DEFAULT_RUSTC = Path("/home/scott/.cargo/bin/rustc")
COMMANDS = ("check", "test", "build", "bench")
NAMESPACE = "lfm3-cargo-targets"


class CargoCapacityBlocked(RuntimeError):
    """An SSD-targeted Cargo job is not permitted to start."""


def _device(path):
    return os.stat(path).st_dev


def _available_bytes(path):
    data = os.statvfs(path)
    return data.f_bavail * data.f_frsize


def _safe_path(path):
    p = Path(path).expanduser().absolute()
    if ".." in p.parts or any(item.is_symlink() for item in (p, *p.parents)):
        raise ValueError("parent traversal or symlink component is prohibited")
    return p


def _finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("capacity thresholds must be finite nonnegative numbers")
    return value


def _existing_parent(path):
    p = path
    while not p.exists():
        if p == p.parent:
            raise ValueError("no existing parent for destination")
        p = p.parent
    if not p.is_dir():
        raise ValueError("destination is under a non-directory")
    return p


def plan(repo, ssd_mount=DEFAULT_SSD, *, command="check", release=False,
         root_floor_gib=20, ssd_floor_gib=80, planned_write_gib=5,
         ssd_reserve_gib=32, cargo_binary=DEFAULT_CARGO,
         rustc_binary=DEFAULT_RUSTC):
    for n in (root_floor_gib, ssd_floor_gib, planned_write_gib, ssd_reserve_gib):
        _finite_nonnegative(n)
    if command not in COMMANDS or type(release) is not bool:
        raise ValueError("unsupported Cargo command or release mode")
    repo, mount = _safe_path(repo), _safe_path(ssd_mount)
    if not repo.is_dir() or not (repo / "Cargo.toml").is_file():
        raise ValueError("repository must have a local Cargo.toml")
    if (repo / "Cargo.toml").is_symlink():
        raise ValueError("manifest cannot be a symlink")
    if not mount.is_dir() or not mount.is_mount():
        raise ValueError("SSD must be an actual mounted filesystem")
    root_device, ssd_device = _device(Path("/")), _device(mount)
    if root_device == ssd_device:
        raise ValueError("SSD target is on the same filesystem as root")
    digest = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.name).strip("-.") or "project"
    target = _safe_path(mount / NAMESPACE / f"{label}-{digest}" / "target")
    parent = _existing_parent(target)
    if _device(parent) != ssd_device:
        raise ValueError("resolved Cargo target path is not on SSD device")
    root_available = _available_bytes(Path("/"))
    ssd_available = _available_bytes(mount)
    required_ssd = max(ssd_floor_gib, planned_write_gib + ssd_reserve_gib)
    ready = (root_available >= root_floor_gib * GIB and
             ssd_available >= required_ssd * GIB)
    executable = Path(cargo_binary).expanduser().absolute()
    compiler = Path(rustc_binary).expanduser().absolute()
    return {
        "evidence_class": "non-destructive opt-in Cargo SSD target plan",
        "repo": str(repo),
        "manifest": str(repo / "Cargo.toml"),
        "ssd_mount": str(mount),
        "ssd_device": ssd_device,
        "root_device": root_device,
        "cargo_target_dir": str(target),
        "cargo_incremental": "0",
        "ssd_tmpdir": str(target.parent / "tmp"),
        "cargo_binary": str(executable),
        "rustc_binary": str(compiler),
        "cargo_command": [str(executable), command, "--locked", "--offline",
                          "--manifest-path", str(repo / "Cargo.toml")] +
                         (["--release"] if release else []),
        "root_available_gib": round(root_available / GIB, 3),
        "ssd_available_gib": round(ssd_available / GIB, 3),
        "required_root_free_gib": root_floor_gib,
        "required_ssd_free_gib": required_ssd,
        "can_launch": ready,
        "execution_opt_in_required": True,
        "directories_created": False,
        "other_builds_modified": False,
        "space_reserved": False,
    }


def run_explicit(job):
    """Run only an opted-in, checked Cargo command with SSD-local artifacts."""
    if not job["can_launch"]:
        raise CargoCapacityBlocked("root or SSD available space is below policy floor")
    executable = Path(job["cargo_binary"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Cargo executable must exist and be executable")
    compiler = Path(job["rustc_binary"])
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise ValueError("Rust compiler must exist and be executable")
    target = _safe_path(job["cargo_target_dir"])
    temp = _safe_path(job["ssd_tmpdir"])
    mount = _safe_path(job["ssd_mount"])
    if not mount.is_mount() or _device(mount) != job["ssd_device"]:
        raise CargoCapacityBlocked("SSD mount identity changed since planning")
    # Recheck both devices and thresholds immediately before directory creation.
    root_free = _available_bytes(Path("/"))
    ssd_free = _available_bytes(mount)
    if (root_free < job["required_root_free_gib"] * GIB
            or ssd_free < job["required_ssd_free_gib"] * GIB):
        raise CargoCapacityBlocked("available capacity changed since planning")
    if _device(_existing_parent(target)) != job["ssd_device"]:
        raise CargoCapacityBlocked("target parent no longer resides on SSD")
    # Target namespace is isolated per repository. No existing root target touched.
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (_device(target) != job["ssd_device"] or
            _device(temp) != job["ssd_device"] or
            any(x.is_symlink() for x in (target, *target.parents))):
        raise CargoCapacityBlocked("new build destination failed SSD identity check")
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    env["CARGO_INCREMENTAL"] = "0"
    env["RUSTC"] = str(compiler)
    env["TMPDIR"] = str(temp)
    return subprocess.run(
        job["cargo_command"], cwd=job["repo"], env=env, check=False,
    ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ssd-mount", type=Path, default=DEFAULT_SSD)
    parser.add_argument("--cargo-binary", type=Path, default=DEFAULT_CARGO)
    parser.add_argument("--rustc-binary", type=Path, default=DEFAULT_RUSTC)
    parser.add_argument("--command", choices=COMMANDS, default="check")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--root-floor-gib", type=float, default=20)
    parser.add_argument("--ssd-floor-gib", type=float, default=80)
    parser.add_argument("--ssd-reserve-gib", type=float, default=32)
    parser.add_argument("--planned-write-gib", type=float, default=5)
    parser.add_argument("--execute", action="store_true",
                        help="run Cargo after rechecking root and SSD capacity")
    args = parser.parse_args()
    job = plan(
        args.repo, args.ssd_mount, command=args.command,
        release=args.release, root_floor_gib=args.root_floor_gib,
        ssd_floor_gib=args.ssd_floor_gib,
        planned_write_gib=args.planned_write_gib,
        ssd_reserve_gib=args.ssd_reserve_gib,
        cargo_binary=args.cargo_binary, rustc_binary=args.rustc_binary,
    )
    if not args.execute:
        print(json.dumps(job, indent=2))
        raise SystemExit(0 if job["can_launch"] else 2)
    code = run_explicit(job)
    print(json.dumps({
        "evidence_class": "opt-in isolated Cargo command status",
        "exit_status": code,
        "cargo_target_dir": job["cargo_target_dir"],
        "root_global_settings_changed": False,
        "other_running_builds_modified": False,
    }, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
