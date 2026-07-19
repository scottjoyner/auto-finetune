"""CLI entrypoint: `python -m src.cli <command>`."""
from __future__ import annotations

import sys

from src.config import load
from src.train import TrainError


def _parse_source(argv: list[str]) -> str | None:
    """Extract --source=<name> from argv (hermes|opencode)."""
    for a in argv:
        if a.startswith("--source="):
            return a.split("=", 1)[1]
    return None


def _parse_label(argv: list[str]) -> str | None:
    """Extract --label=<name> from argv."""
    for a in argv:
        if a.startswith("--label="):
            return a.split("=", 1)[1]
    return None


def _parse_int_flag(argv: list[str], name: str) -> int | None:
    """Extract --name=<int> from argv (e.g. --max-examples=100)."""
    for a in argv:
        if a.startswith(f"{name}="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    cfg = load()
    source = _parse_source(argv)
    label = _parse_label(argv)
    try:
        if cmd == "extract":
            from src.extract_opencode import main as run
            cfg.ensure_dirs()
            return run(cfg, label=label)
        if cmd == "hermes":
            from src.extract_hermes import main as run
            cfg.ensure_dirs()
            return run(cfg)
        if cmd == "clean":
            from src.clean import main as run
            return run(cfg, label=label)
        if cmd == "format":
            from src.format_dataset import main as run
            # --all-split: produce hermes-only, opencode-only, and merged
            if "--all-split" in argv:
                n = 0
                for s in ("hermes", "opencode", None):
                    n += run(cfg, source=s, label=label)
                return n
            return run(cfg, source=source, label=label)
        if cmd == "train":
            from src.train import main as run
            dry = "--dry-run" in argv
            max_ex = _parse_int_flag(argv, "--max-examples")
            return run(cfg, dry_run=dry, source=source, max_examples=max_ex)
        if cmd == "all":
            from src.extract_opencode import main as run_extract
            from src.extract_hermes import main as run_hermes
            from src.clean import main as run_clean
            from src.format_dataset import main as run_format
            cfg.ensure_dirs()
            run_extract(cfg)
            run_hermes(cfg)
            run_clean(cfg)
            for s in ("hermes", "opencode", None):
                run_format(cfg, source=s)
            print("[all] extraction -> cleaning -> formatting complete. Run `train` on a GPU machine.")
            return 0
    except TrainError as e:
        print(f"[error] {e}")
        return 2
    print(__doc__)
    print("Commands: extract | hermes | clean | format | train | all")
    print("Flags:    --source=hermes|opencode  --label=<name>  --all-split  --dry-run  --max-examples=<n>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
