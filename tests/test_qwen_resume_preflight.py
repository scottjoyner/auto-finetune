"""Read-only resume preflight: small fixtures; no model, torch or GPU load."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import qwen_resume_preflight as p


def fixture(tmp_path: Path, *, resume: bool = False):
    staging = tmp_path / "staging"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src/train.py").write_text(
        "def train(trainer):\n    trainer.train(resume_from_checkpoint=\"ckpt\")\n"
        if resume else
        "def train(trainer):\n    trainer.train()\n"
    )
    (staging / "launch-next.stop").parent.mkdir(parents=True)
    (staging / "launch-next.stop").write_text("hold")
    label = staging / "outputs/checkpoints/qwen3-8b-sft-r1"
    label.mkdir(parents=True)
    for step in (50, 100, 150):
        cp = label / f"checkpoint-{step}"
        cp.mkdir()
        for name in p.MANDATORY:
            (cp / name).write_bytes(("data-for-" + name).encode())
        (cp / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 2000}))
    cp = label / "checkpoint-150"
    base = staging / "models/Qwen3-8B"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "model-00001-of-00002.safetensors").write_bytes(b"shard1")
    (base / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    (base / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.1": "model-00001-of-00002.safetensors",
                       "layer.2": "model-00002-of-00002.safetensors"}
    }))
    (cp / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    provenance = staging / "data/future-runs/mixed-seed42"
    provenance.mkdir(parents=True)
    train = tmp_path / "train.jsonl"
    eval = tmp_path / "eval.jsonl"
    source = tmp_path / "source.jsonl"
    for file in (train, eval, source):
        file.write_text("{}\n")
    (provenance / "partition.json").write_text(json.dumps({
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_path": str(train),
        "eval_path": str(eval),
        "contamination": {"status": "clean", "overlap_count": 0},
    }))
    return staging, repo


def test_realistic_complete_fixture_still_no_restore_claim(tmp_path):
    stage, repo = fixture(tmp_path, resume=True)
    result = p.audit(stage, repo, hash_partition_source=True)
    assert result["status"] == "NO_GO"
    assert result["may_resume"] is False
    assert result["may_copy"] is False
    assert result["may_prune"] is False
    assert result["restore_test_passed"] is False
    assert result["dataset"]["source_hash_matches"] is True
    assert result["reasons"] == []
    assert result["training_source"]["all_training_backends_support_resume"] is True


def test_checkpoint_step_and_required_parts(tmp_path):
    stage, repo = fixture(tmp_path)
    result = p.audit(stage, repo)
    assert result["checkpoint"]["step"] == 150
    assert result["checkpoint"]["global_step"] == 150
    assert "training_code_has_no_resume_entrypoint" in result["reasons"]
    ck = stage / "outputs/checkpoints/qwen3-8b-sft-r1/checkpoint-150"
    (ck / "optimizer.pt").unlink()
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 120}))
    result = p.audit(stage, repo)
    assert set(result["checkpoint"]["missing"]) == {"optimizer.pt"}
    assert "checkpoint_components_missing" in result["reasons"]
    assert "trainer_state_step_mismatch" in result["reasons"]


def test_deleted_or_changed_dataset_never_claims_clean(tmp_path):
    stage, repo = fixture(tmp_path)
    provenance = stage / "data/future-runs/mixed-seed42/partition.json"
    manifest = json.loads(provenance.read_text())
    manifest["contamination"] = {"status": "contaminated", "overlap_count": 1}
    provenance.write_text(json.dumps(manifest))
    result = p.audit(stage, repo)
    assert "dataset_partition_not_clean" in result["reasons"]


def test_dataset_hash_drift_detected_without_loading_weights(tmp_path):
    stage, repo = fixture(tmp_path)
    provenance = stage / "data/future-runs/mixed-seed42/partition.json"
    manifest = json.loads(provenance.read_text())
    Path(manifest["source"]).write_text("modified after partition")
    result = p.audit(stage, repo, hash_partition_source=True)
    assert result["dataset"]["source_hash_matches"] is False
    assert "dataset_source_sha256_unverified" in result["reasons"]


def test_missing_or_symlinked_checkpoint_must_fail(tmp_path):
    stage, repo = fixture(tmp_path)
    cp = stage / "outputs/checkpoints/qwen3-8b-sft-r1/checkpoint-150"
    (cp / "optimizer.pt").unlink()
    (cp / "optimizer.pt").symlink_to(cp.parent / "checkpoint-100" / "optimizer.pt")
    assert "checkpoint_components_missing" in p.audit(stage, repo)["reasons"]


def test_base_model_missing_indexed_shard_not_ready(tmp_path):
    stage, repo = fixture(tmp_path)
    model = stage / "models/Qwen3-8B"
    (model / "model-00002-of-00002.safetensors").unlink()
    result = p.audit(stage, repo)
    assert result["base_model"]["present_shards"] == 1
    assert "base_model_structure_unverified" in result["reasons"]


def test_train_invocations_detect_both_backends(tmp_path):
    path = tmp_path / "train.py"
    path.write_text(
        "def a(trainer):\n  trainer.train()\n"
        "def b(trainer):\n  trainer.train(resume_from_checkpoint=\"x\")\n"
        "def unrelated(obj):\n  obj.train()\n"
    )
    assert p.train_invocations(path) == (2, 1)
    path.write_text("def a(trainer):\n  trainer.train(\n")
    assert p.train_invocations(path) == (0, 0)


def test_training_hold_absent_is_reason_not_permission(tmp_path):
    stage, repo = fixture(tmp_path, resume=True)
    (stage / "launch-next.stop").unlink()
    result = p.audit(stage, repo)
    assert "recovery_stop_file_missing" in result["reasons"]
    assert result["may_train"] is False
    assert result["may_resume"] is False


def test_missing_source_files_never_ready(tmp_path):
    stage, repo = fixture(tmp_path)
    manifest = json.loads((stage / "data/future-runs/mixed-seed42/partition.json").read_text())
    Path(manifest["train_path"]).unlink()
    assert "dataset_partition_files_missing" in p.audit(stage, repo)["reasons"]


def test_no_checkpoint_readily_selected_from_contaminated_label(tmp_path):
    stage, repo = fixture(tmp_path)
    label = stage / "outputs/checkpoints/qwen3-8b-sft-r1-ABORTED-contaminated"
    label.mkdir(parents=True)
    (label / "checkpoint-900").mkdir()
    assert p.audit(stage, repo)["checkpoint"]["step"] == 150
