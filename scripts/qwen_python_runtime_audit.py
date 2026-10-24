#!/usr/bin/env python3
"""Read-only Python trainer venv drift audit; no package imports or repairs.

The probe launches only the interpreter with -I -S (isolated, no site),
never imports Torch, modifies symlinks, runs pip or loads pickle files.
A seemingly compatible interpreter is NOT training/restore admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat
import subprocess

VENV = Path("/media/scott/data/finetune-venv")
NEEDED = (
    "torch","transformers","peft","trl",
    "accelerate","datasets","safetensors","tokenizers",
)
CFG_MAX_BYTES = 16 * 1024
PROBE = (
    "import sys,json;print(json.dumps({"
    "'major':sys.version_info.major,"
    "'minor':sys.version_info.minor,"
    "'micro':sys.version_info.micro,"
    "'executable':sys.executable"
    "}))"
)


def safe_file(path: Path) -> bool:
    try:
        state = path.lstat()
        return stat.S_ISREG(state.st_mode) and state.st_nlink == 1
    except OSError:
        return False


def config(path: Path) -> dict:
    if not safe_file(path) or path.stat().st_size > CFG_MAX_BYTES:
        return {}
    try:
        contents = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeError,OSError):
        return {}
    found = {}
    for line in contents:
        key,sep,value=line.partition("=")
        if sep:
            key=key.strip().lower()
            if key not in found:
                found[key]=value.strip()
    return found


def probe_interpreter(executable: Path) -> dict:
    if not executable.is_file():
        return {}
    try:
        proc=subprocess.run(
            [str(executable),"-I","-S","-c",PROBE],
            capture_output=True,text=True,check=False,timeout=6,
        )
        if proc.returncode!=0 or len(proc.stdout)>2048:
            return {}
        result=json.loads(proc.stdout)
        if (not isinstance(result,dict)
                or type(result.get("major")) is not int
                or type(result.get("minor")) is not int):
            return {}
        return result
    except (OSError,ValueError,subprocess.TimeoutExpired,UnicodeError):
        return {}


def packages_present(site: Path) -> list[str]:
    result=[]
    for name in NEEDED:
        package = site/name.replace("-","_")
        if package.is_dir() and not package.is_symlink():
            result.append(name)
    return result


def audit(venv: Path, *, runtime: dict | None = None) -> dict:
    problems=[]
    cfg=config(venv/"pyvenv.cfg")
    raw=cfg.get("version")
    match=re.fullmatch(r"(\d+)\.(\d+)\.(\d+)",raw or "")
    if match:
        major,minor,patch=map(int,match.groups())
        expected=f"{major}.{minor}"
    else:
        expected=None
        problems.append("venv_creation_interpreter_unverified")
    original=cfg.get("executable")
    original_path=Path(original) if isinstance(original,str) and original.startswith("/") else None
    original_present=bool(original_path and original_path.is_file())
    if not original_present:
        problems.append("original_venv_interpreter_missing")
    interpreter=venv/"bin/python"
    actual=runtime if runtime is not None else probe_interpreter(interpreter)
    if not actual:
        problems.append("active_interpreter_unavailable")
        observed=None
    else:
        observed=f"{actual['major']}.{actual['minor']}"
        if expected is None or observed!=expected:
            problems.append("interpreter_minor_version_drift")
    old_site=venv/"lib"/f"python{expected}"/"site-packages" if expected else None
    live_site=venv/"lib"/f"python{observed}"/"site-packages" if observed else None
    old_packages=packages_present(old_site) if old_site else []
    live_packages=packages_present(live_site) if live_site else []
    if old_site is not None and old_site.is_dir() and len(old_packages)>0 and old_site!=live_site:
        problems.append("training_dependencies_stranded_in_prior_site_packages")
    if len(live_packages)<len(NEEDED):
        problems.append("active_environment_missing_required_training_packages")
    old_dist_count=sum(1 for p in old_site.glob("*.dist-info") if p.is_dir()) if old_site and old_site.is_dir() else 0
    return {
        "status":"RUNTIME_RESTORE_HOLD",
        "read_only":True,
        "venv_path":str(venv),
        "created_python_version":raw,
        "created_python_minor":expected,
        "active_python_version":(
            f"{actual['major']}.{actual['minor']}.{actual.get('micro',0)}"
            if actual else None
        ),
        "active_python_minor":observed,
        "original_interpreter_path":str(original_path) if original_path else None,
        "original_interpreter_exists":original_present,
        "prior_site_metadata_distribution_count":old_dist_count,
        "required_training_packages_in_old_site":sorted(old_packages),
        "required_training_packages_in_active_site":sorted(live_packages),
        "issues":sorted(set(problems)),
        "interpreter_and_site_structure_compatible":not problems,
        "model_loaded":False,
        "trainer_restore_performed":False,
        "may_create_or_modify_environment":False,
        "may_train":False,
        "may_resume":False,
        "may_prune":False,
        "may_release_source":False,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv",type=Path,default=VENV)
    args=parser.parse_args()
    print(json.dumps(audit(args.venv),sort_keys=True))
    return 78


if __name__=="__main__":
    raise SystemExit(main())
