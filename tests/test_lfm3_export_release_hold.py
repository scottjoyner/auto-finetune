"""No model/network/file access: the release gate blocks every export path."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import model_export

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fmt", ["merged", "gguf", "onnx"])
@pytest.mark.parametrize("base", [
    "LiquidAI/LFM3-1.7B", "LiquidAI/lfm3-1.7b",
    "Qwen/Qwen3-8B", "other/unknown", "misleading-alias",
])
def test_exports_rejected_for_every_declared_model_family(fmt, base, tmp_path):
    target = str(tmp_path / "MUST_NOT_WRITE")
    source = str(tmp_path / "MUST_NOT_READ")
    if fmt == "merged":
        result = model_export.export_merged(source, target, base)
    elif fmt == "onnx":
        result = model_export.export_onnx(source, target, base)
    else:
        result = model_export.export_gguf(source, target, "q8_0")
    assert result.success is False
    assert result.size_bytes == 0
    assert result.duration_seconds == 0.0
    assert result.error == "trusted_release_approval_not_implemented"
    assert result.output_path == target
    assert not Path(target).exists()
    assert not Path(source).exists()


@pytest.mark.parametrize("fmt", ["merged", "gguf", "onnx"])
def test_existing_output_dir_and_contents_unchanged(fmt, tmp_path):
    target = tmp_path / "preexisting"
    target.mkdir()
    sentinel = target / "model.safetensors"
    sentinel.write_bytes(b"DO_NOT_CHANGE")
    source = str(tmp_path / "candidate")
    before = sentinel.read_bytes()
    if fmt == "merged":
        r = model_export.export_merged(source, str(target))
    elif fmt == "gguf":
        r = model_export.export_gguf(source, str(target))
    else:
        r = model_export.export_onnx(source, str(target))
    assert not r.success and r.error == model_export.DENIAL_REASON
    assert sentinel.read_bytes() == before
    assert list(target.iterdir()) == [sentinel]


@pytest.mark.parametrize("quantize", ["q8_0", "q4_k_m", "f16"])
def test_gguf_formats_are_also_denied(quantize, tmp_path):
    path = str(tmp_path / "candidate.gguf")
    r = model_export.export_gguf("nonexistent", path, quantize)
    assert r.format == f"gguf-{quantize}"
    assert not r.success
    assert not Path(path).exists()


def test_export_result_preserves_compatibility_fields():
    r = model_export.ExportResult(format="merged", output_path="something")
    assert dataclasses.asdict(r) == {
        "format": "merged", "output_path": "something",
        "size_bytes": 0, "success": False,
        "error": model_export.DENIAL_REASON, "duration_seconds": 0.0,
    }


def test_release_gate_does_not_probe_paths_or_invoke_conversion(monkeypatch, tmp_path):
    def forbid(*args, **kwargs):
        raise AssertionError("forbidden IO or subprocess action")
    with monkeypatch.context() as m:
        m.setattr("subprocess.run", forbid)
        m.setattr("os.makedirs", forbid)
        m.setattr("os.path.exists", forbid)
        m.setattr("os.path.getsize", forbid)
        m.setattr("builtins.open", forbid)
        a = model_export.export_merged("sensitive", str(tmp_path / "merged"))
        b = model_export.export_gguf("sensitive", str(tmp_path / "file.gguf"))
        c = model_export.export_onnx("sensitive", str(tmp_path / "onnx"))
    assert not any((a.success, b.success, c.success))


def test_export_module_import_never_requires_ml_runtime():
    snippet = """
import builtins
import sys
real = builtins.__import__
def check(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name == "transformers" or name.startswith("transformers.") or name == "peft" or name.startswith("peft."):
        raise AssertionError("ML runtime imported before export admission")
    return real(name, *args, **kwargs)
builtins.__import__ = check
from src import model_export
assert model_export.export_merged("src", "dst").success is False
assert model_export.export_gguf("src", "dst").success is False
assert model_export.export_onnx("src", "dst").success is False
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
print("NO_MODEL_IMPORTS")
"""
    result = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT,
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert "NO_MODEL_IMPORTS" in result.stdout


@pytest.mark.parametrize("fmt", ["merged", "gguf", "onnx"])
def test_cli_blocks_before_read_or_write_and_does_not_echo_paths(fmt, tmp_path):
    private_name = "SECRET_NEVER_LOG_THIS_PROMPT"
    input_path = tmp_path / private_name
    output_path = tmp_path / f"model-{fmt}"
    args = [sys.executable, "-m", "src.model_export",
            "--checkpoint", str(input_path), "--output", str(output_path),
            "--format", fmt, "--base-model", "LiquidAI/LFM3-1.7B"]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=8)
    assert result.returncode == 78
    data = json.loads(result.stdout)
    assert data == {
        "schema": model_export.SCHEMA,
        "status": "blocked",
        "reason": model_export.DENIAL_REASON,
        "model_export_permitted": False,
        "artifact_written": False,
        "weights_loaded": False,
        "converter_launched": False,
    }
    assert not input_path.exists() and not output_path.exists()
    assert private_name not in result.stdout + result.stderr
    assert "LiquidAI/LFM3-1.7B" not in result.stdout + result.stderr


def test_cli_missing_required_args_never_creates_model_artifacts(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.model_export"],
                       cwd=ROOT, capture_output=True, text=True, timeout=8)
    assert r.returncode == 2
    assert not list(tmp_path.iterdir())


def test_cli_invalid_format_is_rejected_by_argparse(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.model_export",
                        "--checkpoint", "fake", "--output", str(tmp_path / "a"),
                        "--format", "raw-weights"],
                       cwd=ROOT, capture_output=True, text=True, timeout=8)
    assert r.returncode == 2
    assert not list(tmp_path.iterdir())


def test_cli_help_never_loads_model(tmp_path):
    r = subprocess.run([sys.executable, "-m", "src.model_export", "--help"],
                       cwd=ROOT, capture_output=True, text=True, timeout=8)
    assert r.returncode == 0
    assert "--format" in r.stdout
    assert not list(tmp_path.iterdir())


def test_environment_bypass_flags_do_not_enable_export(monkeypatch, tmp_path):
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setenv("ENABLE_MODEL_EXPORT", "1")
    assert model_export.main(["--checkpoint", "private",
                              "--output", str(tmp_path / "blocked")]) == 78
    assert not (tmp_path / "blocked").exists()


def test_traversal_and_symlink_target_are_never_modified(tmp_path):
    target = tmp_path / "target"
    target.write_text("PRESERVE_THIS")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    r = model_export.export_merged("fake", str(alias))
    assert r.success is False
    assert target.read_text() == "PRESERVE_THIS"
    assert alias.is_symlink()


@pytest.mark.parametrize("func_name", ["export_merged", "export_gguf", "export_onnx"])
def test_source_is_never_statd_even_with_missing_checkpoint(monkeypatch, tmp_path, func_name):
    spy = Mock(side_effect=AssertionError("stat must not run"))
    with monkeypatch.context() as m:
        m.setattr("os.stat", spy)
        result = getattr(model_export, func_name)("nonexistent", str(tmp_path / "output"))
    assert not result.success
    spy.assert_not_called()


@pytest.mark.parametrize("fmt", ["merged", "gguf", "onnx"])
def test_main_cli_dispatcher_respects_export_hold(fmt, tmp_path, capsys):
    from src.cli import _dispatch
    from src.config import Config

    output = tmp_path / f"{fmt}.artifact"
    cfg = Config(raw={"paths": {"analysis_dir": str(tmp_path)}})
    r = _dispatch(["cli", "export", "--checkpoint", "absent",
                   "--output", str(output), "--format", fmt], cfg=cfg)
    assert r == 78
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_export_permitted"] is False
    assert not output.exists()


def test_main_cli_missing_export_approval_never_degrades_to_success(capsys):
    from src.cli import _dispatch
    from src.config import Config

    r = _dispatch(["cli", "export", "--checkpoint", "fake",
                   "--output", "private-output"], cfg=Config(raw={}))
    assert r == 78
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
