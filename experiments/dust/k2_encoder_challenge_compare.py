"""Bounded second-encoder CPU challenge before any additional private scan.

Accepts only local frozen model snapshots, no network downloads. E5 families
require a 'query: ' prefix for symmetric-text similarity; the prefix is an
explicit recorded experimental condition. This is exploratory model
comparison, NOT threshold refitting and NOT classifier fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .k2_semantic_challenge_contract import PAIRS, evaluate_challenge
from .k2_offline_semantic_audit import encode, full_sha


def run_challenge(model_dir: Path, *, prefix: str) -> dict:
    if prefix not in ("", "query: "):
        raise ValueError("unknown model input-prefix condition")
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("must be an existing local model snapshot")
    config = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("local model snapshot is incomplete")
    snippets = [prefix + phrase for a, b, _ in PAIRS for phrase in (a,b)]
    matrix = encode(model_dir, snippets)
    scores = [float(matrix[i] @ matrix[i+1])
              for i in range(0,len(snippets),2)]
    result = evaluate_challenge(scores)
    positive_scores=[s for s, (_,_,label) in zip(scores, PAIRS) if label]
    negative_scores=[s for s, (_,_,label) in zip(scores, PAIRS) if not label]
    return {
        "schema":"auto-finetune.dust-k2-local-encoder-comparison.v1",
        "mode":"FROZEN_CHALLENGE_ONLY_NO_PRIVATE_CORPORA",
        "model_config_sha256":full_sha(config),
        "model_weights_sha256":full_sha(weights),
        "embedding_input_prefix":prefix,
        "mean_positive_cosine":sum(positive_scores)/len(positive_scores),
        "mean_hard_negative_cosine":sum(negative_scores)/len(negative_scores),
        "min_positive_cosine":min(positive_scores),
        "max_hard_negative_cosine":max(negative_scores),
        "frozen_challenge":result,
        "model_threshold_posthoc_tuning_authorized":False,
        "new_classifier_training_authorized":False,
        "real_semantic_source_independence_certified":False,
        "commercial_dataset_reuse_approved":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenge-only",action="store_true")
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--prefix",choices=("", "query: "),default="query: ")
    p.add_argument("--private-output",required=True,type=Path)
    args=p.parse_args(argv)
    if not args.challenge_only:
        p.error("explicit frozen challenge-only flag required")
    dest=args.private_output
    if (dest.exists() or dest.is_symlink() or
            not dest.parent.is_dir() or dest.parent.stat().st_mode & 0o077):
        p.error("exclusive mode-600 output in private mode-700 directory")
    os.umask(0o077)
    result=run_challenge(args.model_dir,prefix=args.prefix)
    with dest.open("x",encoding="utf-8") as fd:
        fd.write(json.dumps(result,sort_keys=True,indent=2)+"\n")
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
