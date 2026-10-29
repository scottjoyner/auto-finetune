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
        def from_pretrained(cls, path, *, local_files_only, trust_remote_code):
            assert path == location and local_files_only and trust_remote_code
            return cls()

    class Model:
        def __init__(self, *args, **kwargs):
            raise AssertionError("random constructor must never run")

        @classmethod
        def from_pretrained(
            cls, path, *, config, local_files_only, trust_remote_code, dtype, output_loading_info
        ):
            assert path == location and local_files_only and trust_remote_code
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
            return object(), {
                "missing_keys": ["model.layer.weight"],
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            }

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
    assert "lfm3_dir = verify_official_lfm3_snapshot(" in code
    assert "lfm3_dir, AutoConfig, AutoModelForCausalLM" in code
    assert "from modeling_lfm3 import Lfm3ForCausalLM" not in code


def test_unavailable_pretrained_api_is_explicitly_refused(tmp_path):
    location = single(tmp_path)

    class Config:
        model_type = "lfm3"

    class Model:
        pass

    with pytest.raises(LFM3CheckpointError, match="loader_api_incompatible"):
        load_pretrained_lfm3(location, Config, Model, "bf16")


def test_snapshot_pins_admit_exact_files_and_refuse_tampering(tmp_path, monkeypatch):
    import hashlib

    from src import lfm3_pretrained_guard as guard

    location = single(tmp_path)
    for filename in ("modeling_lfm3.py", "configuration_lfm3.py", "lfm3_conv.py", "tokenizer.json"):
        (tmp_path / filename).write_bytes(b"synthetic example")
    names = (
        "config.json",
        "model.safetensors",
        "modeling_lfm3.py",
        "configuration_lfm3.py",
        "lfm3_conv.py",
        "tokenizer.json",
    )
    pins = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names}
    monkeypatch.setattr(guard, "OFFICIAL_LFM3_SHA256", pins)
    monkeypatch.setattr(guard, "audit_tensor_manifest", lambda path: {"synthetic_only": True})
    assert guard.verify_official_lfm3_snapshot(location) == location
    (tmp_path / "modeling_lfm3.py").write_bytes(b"modified code")
    with pytest.raises(LFM3CheckpointError, match="revision_mismatch:modeling_lfm3.py"):
        guard.verify_official_lfm3_snapshot(location)


def test_missing_pinned_modeling_file_refused(tmp_path, monkeypatch):
    from src import lfm3_pretrained_guard as guard

    location = single(tmp_path)
    monkeypatch.setattr(guard, "OFFICIAL_LFM3_SHA256", {"modeling_lfm3.py": "0" * 64})
    with pytest.raises(LFM3CheckpointError, match="pinned_file_missing"):
        guard.verify_official_lfm3_snapshot(location)


def test_snapshot_rejects_invalid_tensor_header(tmp_path, monkeypatch):
    import hashlib

    from src import lfm3_pretrained_guard as guard

    location = single(tmp_path)
    pins = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors")
    }
    monkeypatch.setattr(guard, "OFFICIAL_LFM3_SHA256", pins)
    with pytest.raises(LFM3CheckpointError, match="tensor_manifest_invalid"):
        guard.verify_official_lfm3_snapshot(location)


def test_empty_set_loading_info_is_valid(tmp_path):
    location = single(tmp_path)
    model = object()

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model, {
                "missing_keys": set(),
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            }

    assert load_pretrained_lfm3(location, Config, Model, "bf16") is model


def test_incomplete_loading_report_is_refused(tmp_path):
    location = single(tmp_path)

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object(), {"missing_keys": set()}

    with pytest.raises(LFM3CheckpointError, match="loading_report_invalid"):
        load_pretrained_lfm3(location, Config, Model, "bf16")


def test_verified_legacy_conv_layer_remap_restored(tmp_path):
    from types import SimpleNamespace

    location = single(tmp_path)
    original = ["conv"] * 13 + ["full_attention"] * 6 + ["sliding_attention"] * 5
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "lfm3", "layer_types": original})
    )

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(
                model_type="lfm3",
                layer_types=["linear_attention" if x == "conv" else x for x in original],
            )

    class Model:
        @classmethod
        def from_pretrained(cls, *args, config, **kwargs):
            return SimpleNamespace(config=config), {
                "missing_keys": set(),
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            }

    result = load_pretrained_lfm3(location, Config, Model, "bf16")
    assert result.config.layer_types == original


def test_unrecognized_layer_mapping_stays_blocked(tmp_path):
    from types import SimpleNamespace

    location = single(tmp_path)
    original = ["conv"] * 13 + ["full_attention"] * 6 + ["sliding_attention"] * 5
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "lfm3", "layer_types": original})
    )

    class Config:
        model_type = "lfm3"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(model_type="lfm3", layer_types=["unknown"] * 24)

    class Model:
        @classmethod
        def from_pretrained(cls, *args, config, **kwargs):
            return SimpleNamespace(config=config), {
                "missing_keys": set(),
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            }

    with pytest.raises(LFM3CheckpointError, match="unexpected_runtime_layer_remap"):
        load_pretrained_lfm3(location, Config, Model, "bf16")