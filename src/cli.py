"""CLI entrypoint: `python -m src.cli <command>`."""
from __future__ import annotations

import sys

from src.config import load
from src.train import TrainError


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    cfg = load()
    try:
        if cmd == "extract":
            from src.extract_opencode import main as run
            cfg.ensure_dirs()
            return run(cfg)
        if cmd == "hermes":
            from src.extract_hermes import main as run
            cfg.ensure_dirs()
            return run(cfg)
        if cmd == "clean":
            from src.clean import main as run
            return run(cfg)
        if cmd == "format":
            from src.format_dataset import main as run
            return run(cfg)
        if cmd == "train":
            from src.train import main as run
            dry = "--dry-run" in argv
            return run(cfg, dry_run=dry)
        if cmd == "all":
            from src.extract_opencode import main as run_extract
            from src.extract_hermes import main as run_hermes
            from src.clean import main as run_clean
            from src.format_dataset import main as run_format
            cfg.ensure_dirs()
            run_extract(cfg)
            run_hermes(cfg)
            run_clean(cfg)
            run_format(cfg)
            print("[all] extraction -> cleaning -> formatting complete. Run `train` on a GPU machine.")
            return 0
    except TrainError as e:
        print(f"[error] {e}")
        return 2
    print(__doc__)
    print("Commands: extract | hermes | clean | format | train | all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
