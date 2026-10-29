"""Metadata-only SafeTensors integrity checks for official LFM3-1.7B.

Never maps, deserializes, or allocates parameter tensors.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

MAX_HEADER_BYTES = 8 * 1024 * 1024
EXPECTED_TENSORS = 286
EXPECTED_ELEMENTS = 1_728_801_536
EXPECTED_DATA_BYTES = EXPECTED_ELEMENTS * 2
EXPECTED_LAYERS = frozenset(range(24))
_LAYER = re.compile(r"model\.layers\.(\d+)\.")


class LFM3TensorManifestError(RuntimeError):
    """Malformed or incompatible official LFM3 checkpoint metadata."""


def _unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise LFM3TensorManifestError("duplicate_json_key")
        result[k] = v
    return result


def audit_tensor_manifest(path: str | Path) -> dict:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise LFM3TensorManifestError("missing_or_symlinked_tensor_file")
    size = p.stat().st_size
    if size < 10:
        raise LFM3TensorManifestError("file_too_small")
    try:
        with p.open("rb") as f:
            length_raw = f.read(8)
            if len(length_raw) != 8:
                raise LFM3TensorManifestError("truncated_header_length")
            header_bytes = struct.unpack("<Q", length_raw)[0]
            if not 2 <= header_bytes <= MAX_HEADER_BYTES or header_bytes + 8 > size:
                raise LFM3TensorManifestError("invalid_header_size")
            raw = f.read(header_bytes)
            if len(raw) != header_bytes:
                raise LFM3TensorManifestError("truncated_header")
    except OSError as exc:
        raise LFM3TensorManifestError("unreadable_tensor_file") from exc
    try:
        header = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError) as exc:
        raise LFM3TensorManifestError("invalid_tensor_header") from exc
    if not isinstance(header, dict):
        raise LFM3TensorManifestError("invalid_header_object")
    if "__metadata__" in header and not isinstance(header.pop("__metadata__"), dict):
        raise LFM3TensorManifestError("invalid_header_metadata")
    if len(header) != EXPECTED_TENSORS:
        raise LFM3TensorManifestError("unexpected_tensor_count")
    spans = []
    layer_indices = set()
    total_elements = 0
    gate_count = 0
    for key, value in header.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise LFM3TensorManifestError("invalid_tensor_entry")
        if set(value) != {"dtype", "shape", "data_offsets"} or value["dtype"] != "BF16":
            raise LFM3TensorManifestError("unsupported_tensor_dtype_or_fields")
        shape, offsets = value["shape"], value["data_offsets"]
        if (
            not isinstance(shape, list)
            or not shape
            or not all(type(n) is int and n > 0 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(x) is int and x >= 0 for x in offsets)
        ):
            raise LFM3TensorManifestError("invalid_tensor_shape_or_offsets")
        start, end = offsets
        elements = math.prod(shape)
        if end - start != 2 * elements or end <= start:
            raise LFM3TensorManifestError("tensor_span_size_mismatch")
        if end > size - 8 - header_bytes:
            raise LFM3TensorManifestError("tensor_span_out_of_range")
        total_elements += elements
        spans.append((start, end))
        layer = _LAYER.match(key)
        if layer:
            layer_indices.add(int(layer.group(1)))
        if key.endswith(".self_attn.gate_proj.weight"):
            gate_count += 1
    if total_elements != EXPECTED_ELEMENTS:
        raise LFM3TensorManifestError("unexpected_parameter_count")
    if layer_indices != EXPECTED_LAYERS:
        raise LFM3TensorManifestError("missing_or_extra_transformer_layers")
    if not gate_count:
        raise LFM3TensorManifestError("attention_gates_missing")
    spans.sort()
    expected_start = 0
    for start, end in spans:
        if start != expected_start:
            raise LFM3TensorManifestError("overlapping_or_gapped_tensor_spans")
        expected_start = end
    if expected_start != EXPECTED_DATA_BYTES or expected_start != size - 8 - header_bytes:
        raise LFM3TensorManifestError("unexpected_tensor_data_size")
    return {
        "tensor_count": len(header),
        "parameter_elements": total_elements,
        "data_bytes": expected_start,
        "layer_count": len(layer_indices),
        "attention_gate_tensors": gate_count,
        "dtype": "BF16",
    }