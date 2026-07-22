"""Provenance manifest tests; all inputs are local temporary files."""
from __future__ import annotations

import json
from pathlib import Path

from src.manifest import build_manifest, write_manifest


def test_manifest_records_required_provenance_and_contamination(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "skills" / "ops").mkdir(parents=True)
    (repo / "src" / "x.py").write_text("x = 1\n")
    (repo / "skills" / "ops" / "SKILL.md").write_text("skill\n")
    (repo / "README.md").write_text("docs\n")
    config = repo / "config.yaml"
    config.write_text("name: test\n")
    train = tmp_path / "train.jsonl"
    bench = tmp_path / "bench.jsonl"
    prompt = "create a sufficiently long configuration file"
    train.write_text(json.dumps({"messages": [{"role": "user", "content": prompt}]}) + "\n")
    bench.write_text(json.dumps({"id": "b1", "prompt": prompt}) + "\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"weights")
    state = tmp_path / "trainer_state.json"
    state.write_text(json.dumps({"global_step": 12, "log_history": [{"loss": 1.2}]}))

    manifest = build_manifest(
        repo_root=repo, datasets=[train], models=[model], artifacts=[artifact],
        config_paths=[config], trainer_state_paths=[state], train_path=train,
        bench_path=bench, argv=["cli", "manifest"], environ={"PATH": "/bin"},
    )
    assert manifest["schema_version"] == 1
    for group in ("datasets", "models", "artifacts", "configs", "code", "skills", "docs"):
        assert manifest["inputs"][group]
        assert manifest["inputs"][group][0]["sha256"]
    assert manifest["trainer"]["states"][0]["data"]["global_step"] == 12
    assert manifest["trainer"]["logs"] == [{"loss": 1.2}]
    assert manifest["evaluation"]["contamination"]["status"] == "contaminated"
    assert "revision" in manifest["git"] and "status" in manifest["git"]
    assert manifest["command"]["argv"] == ["cli", "manifest"]
    assert manifest["environment"]["PATH"] == "/bin"


def test_manifest_write_is_atomic_json(tmp_path, monkeypatch):
    out = tmp_path / "manifest.json"
    write_manifest(out, {"ok": True})
    assert json.loads(out.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))
