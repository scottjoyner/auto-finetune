"""CPU-free tests for fail-closed LFM3 pretrained checkpoint loading."""

import json

import pytest

from src.lfm3_pretrained_guard import (
    LFM3CheckpointError,
    load_pretrained_lfm3,
    verify_local_checkpoint,
)


def single(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"lfm3"}')
    (tmp_path / "model.safetensors").write_bytes(b"synthetic test bytes")
    return str(tmp_path)


def shards(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"lfm3"}')
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layer.weight": "model-00001-of-00001.safetensors"}})
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"synthetic test bytes")
    return str(tmp_path)


def test_single_weights_present(tmp_path):
    assert verify_local_checkpoint(single(tmp_path)) == str(tmp_path)


def test_sharded_weights_present(tmp_path):
    assert verify_local_checkpoint(shards(tmp_path)) == str(tmp_path)


def test_missing_weights_refused(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(LFM3CheckpointError, match="weights_missing"):
        verify_local_checkpoint(str(tmp_path))


def test_missing_shard_refused(tmp_path):
    shards(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(LFM3CheckpointError, match="shard_missing"):
        verify_local_checkpoint(str(tmp_path))


def test_unsafe_index_path_refused(tmp_path):
    shards(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../outside.safetensors"}})
    )
    with pytest.raises(LFM3CheckpointError, match="unsafe_path"):
        verify_local_checkpoint(str(tmp_path))


def test_missing_directory_refused(tmp_path):
    with pytest.raises(LFM3CheckpointError, match="directory_missing"):
        verify_local_checkpoint(str(tmp_path / "missing"))


def test_symlink_checkpoint_refused(tmp_path):
    single(tmp_path)
    link = tmp_path.parent / (tmp_path.name + "-symbolic")
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LFM3CheckpointError, match="symlink"):
        verify_local_checkpoint(str(link))


def test_ambiguous_weight_layout_refused(tmp_path):
    single(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(LFM3CheckpointError, match="ambiguous"):
        verify_local_checkpoint(str(tmp_path))


def test_verified_from_pretrained_not_constructor(tmp_path):
    location = single(tmp_path)
    model = object()

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            assert path == location and local_files_only
            return cls()

    class Model:
        def __init__(self, *args, **kwargs):
            raise AssertionError("random constructor must never run")

        @classmethod
        def from_pretrained(cls, path, *, config, local_files_only, dtype, output_loading_info):
            assert path == location and local_files_only
            assert config.model_type == "lfm3" and dtype == "bf16" and output_loading_info
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    assert load_pretrained_lfm3(location, Config, Model, "bf16") is model


def test_missing_state_keys_are_not_accepted(tmp_path):
    location = single(tmp_path)

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object(), {"missing_keys": ["model.layer.weight"]}

    with pytest.raises(LFM3CheckpointError, match="keys_incomplete"):
        load_pretrained_lfm3(location, Config, Model, "bf16")


def test_missing_load_report_refused(tmp_path):
    location = single(tmp_path)

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    with pytest.raises(LFM3CheckpointError, match="loading_report_missing"):
        load_pretrained_lfm3(location, Config, Model, "bf16")


def test_wrong_model_type_refused(tmp_path):
    location = single(tmp_path)

    class Config:
        model_type = "qwen"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("should not load mismatched model")

    with pytest.raises(LFM3CheckpointError, match="model_type_mismatch"):
        load_pretrained_lfm3(location, Config, Model, "bf16")


def test_train_entrypoint_calls_verified_pretrained_loader():
    from pathlib import Path

    code = (Path(__file__).resolve().parents[1] / "src/train.py").read_text()
    assert "model = Lfm3ForCausalLM(config)" not in code
    assert "model = load_pretrained_lfm3(" in code
    assert "lfm3_dir = verify_local_checkpoint(" in code