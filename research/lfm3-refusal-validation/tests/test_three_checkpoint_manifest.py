import json
from pathlib import Path
import pytest
from src.three_checkpoint_eval import checked_arm_paths


def test_missing_model_rejected(tmp_path):
    arms = {name: tmp_path / name for name in
            ["A_original_posttrained", "B_M1_abliterated_smoke", "C_base_pretrained"]}
    with pytest.raises(FileNotFoundError):
        checked_arm_paths(arms)


def test_same_checkpoint_rejected(tmp_path):
    from src.build_abliterated_checkpoint import digest
    d = tmp_path / "shared"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"identical_weights")
    arms = {name: d for name in
            ["A_original_posttrained", "B_M1_abliterated_smoke", "C_base_pretrained"]}
    with pytest.raises(ValueError, match="collision/alias"):
        checked_arm_paths(arms)


def test_bad_provenance_rejected(tmp_path):
    from src.build_abliterated_checkpoint import digest
    arms = {}
    for name, data in [("A_original_posttrained", b"original"),
                       ("B_M1_abliterated_smoke", b"experimental"),
                       ("C_base_pretrained", b"base")]:
        p = tmp_path / name
        p.mkdir()
        (p / "model.safetensors").write_bytes(data)
        arms[name] = p
    (arms["B_M1_abliterated_smoke"] / "RESEARCH_PROVENANCE.json").write_text(
        json.dumps({"source_sha256": "incorrect", "checkpoint_sha256": "alsoincorrect"}))
    with pytest.raises(ValueError, match="parent and output"):
        checked_arm_paths(arms)
