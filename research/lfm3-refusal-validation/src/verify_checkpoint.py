"""Verify the exact 48-matrix M1 edit; fail closed on all other tensor changes."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from src.build_abliterated_checkpoint import (
    DEFAULT_SOURCE, DEFAULT_OUTPUT, DEFAULT_VECTOR, digest,
    orthogonalize, residual_writer_names
)


def verify(source=DEFAULT_SOURCE, target=DEFAULT_OUTPUT, vector=DEFAULT_VECTOR):
    metadata = json.loads((target / "RESEARCH_PROVENANCE.json").read_text())
    original_path, edited_path = source / "model.safetensors", target / "model.safetensors"
    assert digest(original_path) == metadata["source_sha256"], "source checksum"
    assert digest(edited_path) == metadata["checkpoint_sha256"], "target checksum"
    assert digest(vector) == metadata["direction_sha256"], "direction checksum"
    assert (source / "config.json").read_bytes() == (target / "config.json").read_bytes()
    config = json.loads((source / "config.json").read_text())
    expected = set(residual_writer_names(config))
    r = load_file(str(vector))[f"layer_{metadata['direction_layer']}"]
    changed = []
    with safe_open(original_path, framework="pt") as a, safe_open(edited_path, framework="pt") as b:
        left, right = set(a.keys()), set(b.keys())
        assert left == right, "different tensor inventory"
        assert len(expected) == 48 and expected <= left
        for key in sorted(left):
            before = a.get_tensor(key)
            after = b.get_tensor(key)
            if key in expected:
                predicted, _, _ = orthogonalize(before, r, metadata["alpha"])
                assert torch.equal(after, predicted), f"unexpected edit: {key}"
                if not torch.equal(before, after):
                    changed.append(key)
            else:
                assert torch.equal(before, after), f"non-target tensor modified: {key}"
    return {
        "source_sha256": metadata["source_sha256"],
        "target_sha256": metadata["checkpoint_sha256"],
        "source_tensor_count": len(left),
        "selected_residual_writers": len(expected),
        "actual_changed_tensors": len(changed),
        "verified_non_target_unchanged": len(left) - len(expected),
        "exact_projection_verified": True,
        "safety_approved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
