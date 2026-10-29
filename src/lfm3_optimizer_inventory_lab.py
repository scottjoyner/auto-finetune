"""Read-only official LFM3 optimizer-group inventory for an isolated sandbox.

No optimizer allocation, no gradients, no adapter selection or LFM3 training.
Requires external bwrap network+mount isolation (flags alone are insufficient).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_optimizer_partition import plan_full_model_metadata
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment


def inspect(snapshot: Path) -> dict:
    verify_restricted_environment()
    attested = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("optimizer inventory requires pinned isolated Transformers")
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    return plan_full_model_metadata(
        model.named_parameters(),
        model_class=model.__class__.__name__,
        checkpoint_sha256=attested["checkpoint_sha256"],
        explicit_research_review=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", required=True, type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(inspect(args.snapshot), sort_keys=True))
        return 0
    except (ValueError, OSError, ResearchRefused) as exc:
        raise SystemExit("optimizer partition refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
