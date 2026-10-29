"""Fail-closed local LFM3 checkpoint admission. No network or model allocation at import."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


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
    config = config_class.from_pretrained(model_dir, local_files_only=True)
    if getattr(config, "model_type", None) != "lfm3":
        raise LFM3CheckpointError("lfm3_model_type_mismatch")
    result = model_class.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=dtype,
        output_loading_info=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise LFM3CheckpointError("lfm3_loading_report_missing")
    model, details = result
    if not isinstance(details, dict) or not isinstance(details.get("missing_keys"), list):
        raise LFM3CheckpointError("lfm3_loading_report_invalid")
    if any(
        details.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise LFM3CheckpointError("lfm3_weight_keys_incomplete")
    return model