"""Adversarial metadata-only tests; no real tensors or GPU allocations."""

import json
import struct

import pytest

from src import lfm3_tensor_manifest as tm


@pytest.fixture
def spec(monkeypatch):
    monkeypatch.setattr(tm, "EXPECTED_TENSORS", 3)
    monkeypatch.setattr(tm, "EXPECTED_ELEMENTS", 7)
    monkeypatch.setattr(tm, "EXPECTED_DATA_BYTES", 14)
    monkeypatch.setattr(tm, "EXPECTED_LAYERS", frozenset({0, 1}))
    return {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.gate_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [4, 12],
        },
        "model.layers.1.ffn_norm.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [12, 14]},
    }


def write(tmp_path, header, body=b"0" * 14):
    p = tmp_path / "model.safetensors"
    header_bytes = header if isinstance(header, bytes) else json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + body)
    return p


def test_valid_metadata_without_loading_weights(tmp_path, spec):
    outcome = tm.audit_tensor_manifest(write(tmp_path, spec))
    assert outcome == {
        "tensor_count": 3,
        "parameter_elements": 7,
        "data_bytes": 14,
        "layer_count": 2,
        "attention_gate_tensors": 1,
        "dtype": "BF16",
    }


def test_duplicate_json_key_rejected(tmp_path, spec):
    raw = json.dumps(spec)
    key = json.dumps("model.embed_tokens.weight")
    value = json.dumps(spec["model.embed_tokens.weight"])
    raw = "{" + key + ":" + value + "," + raw[1:]
    with pytest.raises(tm.LFM3TensorManifestError, match="duplicate_json_key"):
        tm.audit_tensor_manifest(write(tmp_path, raw.encode()))


def test_missing_layer_rejected(tmp_path, spec):
    spec["model.layers.1.ffn_norm.weight"].update({"data_offsets": [12, 14]})
    spec["model.layers.2.ffn_norm.weight"] = spec.pop("model.layers.1.ffn_norm.weight")
    with pytest.raises(tm.LFM3TensorManifestError, match="layers"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_non_bf16_rejected(tmp_path, spec):
    spec["model.embed_tokens.weight"]["dtype"] = "F16"
    with pytest.raises(tm.LFM3TensorManifestError, match="dtype"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_overlapping_spans_rejected(tmp_path, spec):
    spec["model.layers.0.self_attn.gate_proj.weight"]["data_offsets"] = [2, 10]
    with pytest.raises(tm.LFM3TensorManifestError, match="overlapping_or_gapped"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_out_of_bounds_span_rejected(tmp_path, spec):
    spec["model.layers.1.ffn_norm.weight"]["data_offsets"] = [14, 16]
    with pytest.raises(tm.LFM3TensorManifestError, match="out_of_range"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_bool_dimension_rejected(tmp_path, spec):
    spec["model.embed_tokens.weight"]["shape"] = [True]
    with pytest.raises(tm.LFM3TensorManifestError, match="shape_or_offsets"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_wrong_parameter_count_rejected(tmp_path, spec):
    spec["model.embed_tokens.weight"]["shape"] = [1]
    with pytest.raises(tm.LFM3TensorManifestError, match="span_size_mismatch"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_missing_gates_rejected(tmp_path, spec):
    spec["model.layers.0.feed_forward.w1.weight"] = spec.pop(
        "model.layers.0.self_attn.gate_proj.weight"
    )
    with pytest.raises(tm.LFM3TensorManifestError, match="gates_missing"):
        tm.audit_tensor_manifest(write(tmp_path, spec))


def test_header_size_limit_rejected(tmp_path, spec):
    p = tmp_path / "model.safetensors"
    p.write_bytes(struct.pack("<Q", tm.MAX_HEADER_BYTES + 1) + b"{}")
    with pytest.raises(tm.LFM3TensorManifestError, match="header_size"):
        tm.audit_tensor_manifest(p)


def test_symlink_rejected(tmp_path, spec):
    p = write(tmp_path, spec)
    link = tmp_path / "link.safetensors"
    link.symlink_to(p)
    with pytest.raises(tm.LFM3TensorManifestError, match="symlinked"):
        tm.audit_tensor_manifest(link)


def test_invalid_metadata_rejected(tmp_path, spec):
    spec["__metadata__"] = []
    with pytest.raises(tm.LFM3TensorManifestError, match="header_metadata"):
        tm.audit_tensor_manifest(write(tmp_path, spec))