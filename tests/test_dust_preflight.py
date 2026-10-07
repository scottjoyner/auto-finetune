"""CPU-only regression tests; no model weights, network, GPU or training required."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import dust_preflight as preflight


def model_fixture(tmp_path: Path, complete=True) -> Path:
    root = tmp_path / "k2"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "k2_horizon",
                    "architectures": ["K2HorizonForCausalLM"],
                    "num_hidden_layers": 28, "hidden_size": 1536}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 3},
                    "weight_map": {"a": "model-00000-of-00001.safetensors"}}))
    if complete:
        (root / "model-00000-of-00001.safetensors").write_bytes(b"abc")
    return root


def test_detects_missing_nas_shard_without_treating_index_as_weights(tmp_path):
    info = preflight.model_info(model_fixture(tmp_path, complete=False))
    assert not info["complete"]
    assert info["shard_count"] == 1
    assert info["missing_shards"] == ["model-00000-of-00001.safetensors"]


def test_full_base_and_optional_weight_digest(tmp_path):
    root = model_fixture(tmp_path)
    fast = preflight.model_info(root)
    strict = preflight.model_info(root, hash_weights=True)
    assert fast["complete"] and strict["complete"]
    assert "sha256" not in fast["shards"]["model-00000-of-00001.safetensors"]
    assert strict["shards"]["model-00000-of-00001.safetensors"]["sha256"] == preflight.digest(root / "model-00000-of-00001.safetensors")


def test_adapter_provenance_and_missing_adapter(tmp_path):
    adapter = tmp_path / "v8"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32,
        "target_modules": ["v_proj", "o_proj"]}))
    assert not preflight.adapter_info(adapter)["complete"]
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    info = preflight.adapter_info(adapter, hash_weights=True)
    assert info["complete"]
    assert info["target_modules"] == ["o_proj", "v_proj"]
    assert info["weights"]["adapter_model.safetensors"]["sha256"]


def test_dataset_metadata_only_does_not_include_secret_content(tmp_path):
    corpus = tmp_path / "heldout.jsonl"
    corpus.write_text('{"private":"hidden-needle"}\n\n{"example":1}\n')
    metadata = preflight.dataset_info(corpus)
    assert metadata["nonblank_lines"] == 2
    assert metadata["sha256"] == preflight.digest(corpus)
    assert "hidden-needle" not in json.dumps(metadata)


def test_manifest_is_never_training_authorized(tmp_path, monkeypatch):
    base = model_fixture(tmp_path)
    args = SimpleNamespace(model_dir=base, adapter=[], dataset=[],
                           upstream_dir=None, nas_dir=tmp_path,
                           sha256_weights=False, probe_torch=False)
    report = preflight.build_manifest(args)
    assert report["training_authorized"] is False
    assert not report["blockers"]
    assert report["upstream"]["expected_commit"] == preflight.DUST_PIN


def test_pinned_upstream_revision_and_file_presence(tmp_path, monkeypatch):
    upstream = tmp_path / "dust"
    upstream.mkdir()
    for filename in preflight.UPSTREAM_FILES:
        (upstream / filename).write_text("fixture")
    monkeypatch.setattr(preflight, "git_head", lambda _: (preflight.DUST_PIN, False))
    assert preflight.upstream_info(upstream)["pinned"]
    monkeypatch.setattr(preflight, "git_head", lambda _: ("wrongsha", False))
    assert not preflight.upstream_info(upstream)["pinned"]
    (upstream / "dust.py").unlink()
    assert not preflight.upstream_info(upstream)["all_files_present"]


def test_runtime_avoids_torch_import_without_explicit_probe(monkeypatch):
    info = preflight.runtime_info(False)
    assert info["torch_probed"] is False
    assert "torch_device_available" not in info


def test_cli_creates_new_output_exclusively(tmp_path):
    base = model_fixture(tmp_path)
    out = tmp_path / "manifest.json"
    assert preflight.main(["--model-dir", str(base), "--output", str(out),
                           "--strict-model"]) == 0
    payload = json.loads(out.read_text())
    assert payload["training_authorized"] is False
    with pytest.raises(SystemExit) as error:
        preflight.main(["--model-dir", str(base), "--output", str(out)])
    assert error.value.code == 2


def test_cli_strict_fails_on_missing_weight(tmp_path, capsys):
    base = model_fixture(tmp_path, complete=False)
    assert preflight.main(["--model-dir", str(base), "--strict-model"]) == 2
    captured = json.loads(capsys.readouterr().out)
    assert captured["blockers"] == ["model_incomplete"]
