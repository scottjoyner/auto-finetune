"""Offline, fail-closed LFM3 AdamW/Muon parameter-group planner.

No trainer construction, tensor update, optimizer invocation, or network access.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MAX_PARAMETERS = 20000
SCHEMA = "sensorgraft.lfm3.optimizer-groups.v1"
NAME = re.compile(r"[A-Za-z0-9_.]{1,256}\Z")
HIDDEN = re.compile(
    r"(?:base_model\.model\.)?model\.layers\.(\d+)\."
    r"(?:(?:feed_forward\.(?:w1|w2|w3))|"
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj))|"
    r"(?:conv\.(?:in_proj|out_proj)))\.weight\Z"
)


def _blocked(reason):
    return {
        "schema": SCHEMA,
        "status": "blocked",
        "reason": reason,
        "proposal": None,
        "weight_promotion_permitted": False,
    }


def plan_parameter_groups(parameters, *, mode="lora_only", allow_muon_proposal=False):
    """Suggest categories; NEVER returns actual optimizer objects."""
    if mode not in ("lora_only", "full_finetune_research"):
        return _blocked("unsupported_mode")
    if type(allow_muon_proposal) is not bool:
        return _blocked("invalid_opt_in")
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= MAX_PARAMETERS:
        return _blocked("invalid_parameter_count")
    seen = set()
    muon_candidates = []
    adamw_only = []
    frozen_count = 0
    lora_count = 0
    for p in parameters:
        if not isinstance(p, dict) or set(p) != {"name", "shape", "trainable"}:
            return _blocked("invalid_parameter_record")
        name, shape, trainable = p["name"], p["shape"], p["trainable"]
        if (
            not isinstance(name, str)
            or not NAME.fullmatch(name)
            or name in seen
            or not isinstance(shape, list)
            or not 1 <= len(shape) <= 4
            or any(type(n) is not int or n <= 0 for n in shape)
            or type(trainable) is not bool
        ):
            return _blocked("invalid_or_duplicate_parameter")
        seen.add(name)
        if not trainable:
            frozen_count += 1
            continue
        if ".lora_A." in name or ".lora_B." in name:
            lora_count += 1
        match = HIDDEN.fullmatch(name)
        eligible = (
            mode == "full_finetune_research"
            and allow_muon_proposal
            and len(shape) == 2
            and match is not None
            and int(match.group(1)) in range(24)
        )
        # Do not classify LoRA low-rank factors or the small head gate
        # projections as Muon by dimensionality alone.
        if eligible:
            muon_candidates.append(name)
        else:
            adamw_only.append(name)
    return {
        "schema": SCHEMA,
        "status": "review_required" if muon_candidates else "observed",
        "reason": "muon_experimental_split_needs_review"
        if muon_candidates
        else "adamw_only_until_parameter_compatibility_review",
        "proposal": {
            "muon_matrix_names": sorted(muon_candidates),
            "adamw_names": sorted(adamw_only),
            "frozen_parameter_count": frozen_count,
            "lora_trainable_count": lora_count,
            "mode": mode,
            "requires_independent_holdout_evaluation": bool(muon_candidates),
        },
        "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameters_json", type=Path)
    parser.add_argument(
        "--mode", choices=("lora_only", "full_finetune_research"), default="lora_only"
    )
    parser.add_argument("--allow-muon-proposal", action="store_true")
    args = parser.parse_args()
    if args.parameters_json.stat().st_size > 2_000_000:
        result = _blocked("input_file_too_large")
    else:
        try:
            content = json.loads(args.parameters_json.read_text(encoding="utf-8"))
            result = plan_parameter_groups(
                content,
                mode=args.mode,
                allow_muon_proposal=args.allow_muon_proposal,
            )
        except (OSError, ValueError, UnicodeError):
            result = _blocked("invalid_parameter_file")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()