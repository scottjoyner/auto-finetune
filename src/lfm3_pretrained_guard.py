"""Fail-closed local LFM3 checkpoint admission. No network or model allocation at import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from src.lfm3_tensor_manifest import LFM3TensorManifestError, audit_tensor_manifest


class LFM3CheckpointError(RuntimeError):
    """The requested pretrained checkpoint is absent or not loadable."""


def verify_local_checkpoint(location: str) -> str:
    root = Path(location)
    if root.is_symlink() or not root.is_dir():
        raise LFM3CheckpointError("lfm3_checkpoint_directory_missing_or_symlink")
    config = root / "config.json"
    if config.is_symlink() or not config.is_file() or config.stat().st_size == 0:
        raise LFM3CheckpointError("lfm3_config_missing")
    # Prevent an incomplete shard index, token cache, or config-only checkpoint
    # from being mistaken for pretrained model weights.
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.exists() and index.exists():
        raise LFM3CheckpointError("lfm3_ambiguous_weight_layout")
    if single.exists():
        if single.is_symlink() or not single.is_file() or single.stat().st_size == 0:
            raise LFM3CheckpointError("lfm3_invalid_single_weight_file")
        return str(root)
    if index.is_symlink() or not index.is_file() or index.stat().st_size == 0:
        raise LFM3CheckpointError("lfm3_safetensors_weights_missing")
    if index.stat().st_size > 2_000_000:
        raise LFM3CheckpointError("lfm3_weight_index_oversized")
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise LFM3CheckpointError("lfm3_weight_index_invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise LFM3CheckpointError("lfm3_weight_map_missing")
    shards = set()
    for name in payload["weight_map"].values():
        if (
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or PurePosixPath(name).name != name
            or name in {".", ".."}
        ):
            raise LFM3CheckpointError("lfm3_weight_index_unsafe_path")
        shards.add(name)
    if not 1 <= len(shards) <= 128:
        raise LFM3CheckpointError("lfm3_weight_index_empty_or_excessive")
    for name in shards:
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size == 0:
            raise LFM3CheckpointError("lfm3_weight_shard_missing")
    return str(root)


def load_pretrained_lfm3(location: str, config_class, model_class, dtype):
    """Only return model after checking from_pretrained reported all keys loaded."""
    model_dir = verify_local_checkpoint(location)
    if not callable(getattr(config_class, "from_pretrained", None)) or not callable(
        getattr(model_class, "from_pretrained", None)
    ):
        raise LFM3CheckpointError("lfm3_loader_api_incompatible")
    config = config_class.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if getattr(config, "model_type", None) != "lfm3":
        raise LFM3CheckpointError("lfm3_model_type_mismatch")
    result = model_class.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        output_loading_info=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise LFM3CheckpointError("lfm3_loading_report_missing")
    model, details = result
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if (
        not isinstance(details, dict)
        or any(key not in details for key in fields)
        or any(type(details[key]) not in (list, set, tuple) for key in fields)
    ):
        raise LFM3CheckpointError("lfm3_loading_report_invalid")
    if any(details[key] for key in fields):
        raise LFM3CheckpointError("lfm3_weight_keys_incomplete")

    # Transformers 5.17 remaps legacy "conv" to "linear_attention" at config
    # construction. The pinned official LFM3 model still expects "conv" in its
    # mask mapping. Restore only that exact, known remap after keys are verified.
    try:
        raw_config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise LFM3CheckpointError("lfm3_local_config_unreadable") from exc
    original_layers = raw_config.get("layer_types")
    if original_layers is not None:
        allowed = {"conv", "full_attention", "sliding_attention"}
        if (
            not isinstance(original_layers, list)
            or len(original_layers) != 24
            or any(type(x) is not str or x not in allowed for x in original_layers)
        ):
            raise LFM3CheckpointError("lfm3_unexpected_original_layer_types")
        runtime_layers = getattr(getattr(model, "config", None), "layer_types", None)
        remapped = ["linear_attention" if name == "conv" else name for name in original_layers]
        if runtime_layers != original_layers:
            if runtime_layers != remapped:
                raise LFM3CheckpointError("lfm3_unexpected_runtime_layer_remap")
            model.config.layer_types = list(original_layers)
        if model.config.layer_types != original_layers:
            raise LFM3CheckpointError("lfm3_layer_type_restore_failed")
    return model


# Pinned file digests from LiquidAI/LFM3-1.7B revision 40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd.
# A modified compatibility fork requires separately reviewed source pin updates.
OFFICIAL_LFM3_REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
OFFICIAL_LFM3_SHA256 = {
    "config.json": "8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d",
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
    "tokenizer.json": "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701",
    "tokenizer_config.json": "bf028f813efc6137bc07ed045cafe71820df9a5e26a38e39954706ed1e30a457",
    "chat_template.jinja": "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818",
    "model.safetensors": "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330",
}


def verify_official_lfm3_snapshot(location: str) -> str:
    """Admit only the exact reviewed model, config, tokenizer and source bytes."""
    root = Path(verify_local_checkpoint(location))
    for name, expected in OFFICIAL_LFM3_SHA256.items():
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file():
            raise LFM3CheckpointError("lfm3_pinned_file_missing:" + name)
        h = hashlib.sha256()
        try:
            with candidate.open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(block)
        except OSError as exc:
            raise LFM3CheckpointError("lfm3_pinned_file_unreadable:" + name) from exc
        if h.hexdigest() != expected:
            raise LFM3CheckpointError("lfm3_revision_mismatch:" + name)
    try:
        audit_tensor_manifest(root / "model.safetensors")
    except LFM3TensorManifestError as exc:
        raise LFM3CheckpointError("lfm3_tensor_manifest_invalid:" + str(exc)) from exc
    return str(root)
