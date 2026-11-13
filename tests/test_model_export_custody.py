"""Offline model-export non-overwrite and synthetic-implementation tests.

No model weights or real runtime libraries loaded; all filesystem operations
are limited to pytest tmp_path or mocked tool paths.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src import model_export as exp


def mock_model_libraries(monkeypatch):
    class FakeModel:
        def merge_and_unload(self):
            return self
        def save_pretrained(self, path):
            (Path(path) / "synthetic-model.bin").write_bytes(b"MODEL")
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[0, 1]]}
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeModel(),
    )
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer(),
    )
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeModel(),
    )
    torch = types.ModuleType("torch")
    torch.bfloat16 = "synthetic-bf16"
    def write_onnx(model, args, output, **kwargs):
        Path(output).write_bytes(b"SYNTHETIC_ONNX")
    torch.onnx = types.SimpleNamespace(export=write_onnx)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.parametrize("format_name", ["merged", "onnx"])
def test_existing_output_denied_before_model_import(
        tmp_path, monkeypatch, format_name):
    p = tmp_path / "existing"
    p.mkdir()
    (p / "user-data").write_text("untouched")
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = (exp.export_merged if format_name == "merged" else exp.export_onnx)(
        "synthetic-checkpoint", str(p),
    )
    assert not result.success
    assert "overwrite denied" in result.error
    assert (p / "user-data").read_text() == "untouched"


def test_gguf_preexisting_staging_denied_before_script_search(tmp_path):
    target = tmp_path / "candidate.gguf"
    stage = tmp_path / "candidate.gguf.merged"
    stage.mkdir()
    (stage / "user-data").write_text("keep")
    result = exp.export_gguf("synthetic-checkpoint", str(target))
    assert result.success is False
    assert "overwrite denied" in result.error
    assert (stage / "user-data").read_text() == "keep"


def test_gguf_existing_target_denied_without_cleanup(tmp_path):
    target = tmp_path / "existing.gguf"
    target.write_text("original")
    result = exp.export_gguf("synthetic", str(target))
    assert not result.success
    assert target.read_text() == "original"


def test_missing_llama_conversion_tool_reports_failure(tmp_path, monkeypatch):
    original = exp.os.path.exists
    def fake_exists(path):
        if "convert_hf_to_gguf.py" in str(path):
            return False
        return original(path)
    monkeypatch.setattr(exp.os.path, "exists", fake_exists)
    result = exp.export_gguf("synthetic", str(tmp_path / "fresh.gguf"))
    assert not result.success
    assert "convert script not found" in result.error
    assert not (tmp_path / "fresh.gguf").exists()


def test_synthetic_merged_export_writes_only_tmp(tmp_path, monkeypatch):
    mock_model_libraries(monkeypatch)
    output = tmp_path / "merged"
    result = exp.export_merged("synthetic-adapter", str(output),
                               base_model="synthetic-base")
    assert result.success
    assert result.format == "merged"
    assert result.size_bytes > 0
    assert (output / "synthetic-model.bin").read_bytes() == b"MODEL"


def test_synthetic_onnx_export_writes_only_tmp(tmp_path, monkeypatch):
    mock_model_libraries(monkeypatch)
    output = tmp_path / "onnx"
    result = exp.export_onnx("synthetic-adapter", str(output),
                             base_model="synthetic-base")
    assert result.success
    assert result.format == "onnx"
    assert result.size_bytes >= len(b"SYNTHETIC_ONNX")
    assert (output / "model.onnx").read_bytes() == b"SYNTHETIC_ONNX"


def test_synthetic_lfm3_patching_path_is_observed(tmp_path, monkeypatch):
    mock_model_libraries(monkeypatch)
    calls = []
    monkeypatch.setattr(exp, "is_lfm3_model", lambda name: True)
    monkeypatch.setattr(exp, "patch_transformers_for_lfm3",
                        lambda: calls.append("patch"))
    result = exp.export_merged("fake-adapter", str(tmp_path / "lfm"),
                               base_model="synthetic-lfm3")
    assert result.success
    assert calls == ["patch"]


def test_merge_runtime_import_failure_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    mock_model_libraries(monkeypatch)
    monkeypatch.setitem(sys.modules, "peft", None)
    result = exp.export_merged("synthetic", str(tmp_path / "out"))
    assert result.success is False
    assert result.error
    assert not (tmp_path / "out").exists()


def mock_converter_found(monkeypatch):
    original = exp.os.path.exists
    monkeypatch.setattr(exp.os.path, "exists",
                        lambda path: True if str(path) == "/usr/local/bin/convert_hf_to_gguf.py"
                        else original(path))


def test_gguf_synthetic_merge_failure_prevents_conversion(tmp_path, monkeypatch):
    mock_converter_found(monkeypatch)
    monkeypatch.setattr(
        exp, "export_merged",
        lambda *args, **kwargs: exp.ExportResult(
            format="merged", output_path="synthetic", success=False,
            error="synthetic missing adapter",
        ),
    )
    monkeypatch.setattr(exp.subprocess, "run",
                        lambda *a, **k: pytest.fail("conversion invoked after merge failed"))
    result = exp.export_gguf("synthetic", str(tmp_path / "out.gguf"))
    assert not result.success
    assert "Merge failed" in result.error


@pytest.mark.parametrize("exitcode", [0, 1])
def test_gguf_conversion_with_synthetic_converter(tmp_path, monkeypatch, exitcode):
    mock_converter_found(monkeypatch)
    def fake_merge(checkpoint, merged_dir):
        path = Path(merged_dir)
        path.mkdir()
        (path / "fake.bin").write_bytes(b"model")
        return exp.ExportResult(format="merged", output_path=merged_dir,
                                success=True)
    monkeypatch.setattr(exp, "export_merged", fake_merge)
    target = tmp_path / "fake.gguf"
    def fake_process(cmd, **kwargs):
        assert cmd[0] == sys.executable
        assert cmd[1] == "/usr/local/bin/convert_hf_to_gguf.py"
        assert cmd[-1] == "q4_0"
        if exitcode == 0:
            target.write_bytes(b"FAKE_GGUF")
        return subprocess.CompletedProcess(cmd, exitcode, "", "converter failed")
    monkeypatch.setattr(exp.subprocess, "run", fake_process)
    result = exp.export_gguf("synthetic", str(target), quantize="q4_0")
    assert result.success is (exitcode == 0)
    if exitcode == 0:
        assert result.size_bytes == len(b"FAKE_GGUF")
        assert not (tmp_path / "fake.gguf.merged").exists()
    else:
        assert "converter failed" in result.error


@pytest.mark.parametrize("format_name", ["merged", "gguf", "onnx"])
def test_cli_dispatches_synthetic_result_without_loading_model(
        tmp_path, monkeypatch, capsys, format_name):
    called = []
    def fake_export(*args, **kwargs):
        called.append((args, kwargs))
        return exp.ExportResult(format=format_name, output_path=str(tmp_path),
                                success=True, size_bytes=12345)
    for func in ("export_merged", "export_gguf", "export_onnx"):
        monkeypatch.setattr(exp, func, fake_export)
    assert exp.main(["--checkpoint", "fake", "--output", str(tmp_path / "result"),
                     "--format", format_name]) == 0
    assert called
    assert "Success:" in capsys.readouterr().out


def test_cli_export_failure_status_is_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        exp, "export_merged",
        lambda *a: exp.ExportResult(format="merged", output_path="fake",
                                    success=False, error="synthetic failure"),
    )
    assert exp.main(["--checkpoint", "fake", "--output",
                     str(tmp_path / "out")]) == 1
    assert "Failed: synthetic failure" in capsys.readouterr().out
