#!/usr/bin/env python3
"""Read-only Qwen resume admission report. It never grants training permission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import resume_admission as r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    decision = r.check_live(
        args.checkpoint,
        output_dir=args.output_dir,
        data_path=args.dataset,
        model_name=args.model,
    )
    print(json.dumps({
        "status": "NO_GO", "read_only": True,
        "requested_checkpoint": Path(args.checkpoint).name,
        "resume_admission": "DENIED",
        "reason": decision.reason if not decision.allowed else
                  "independent_launch_authority_not_granted",
        "may_train": False,
        "may_resume": False,
        "may_prune": False,
        "may_release_source": False,
        "restore_test_passed": False,
    }, sort_keys=True))
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
