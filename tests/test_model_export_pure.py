"""Offline, deterministic coverage for ``src.model_export``.

Only the pure / control-flow surface is exercised here:

* the ``ExportResult`` dataclass,
* CLI argument dispatch and return codes in ``main()``,
* the early failure branches of ``export_gguf()`` (missing converter, failed
  merge, failed converter subprocess, successful conversion + cleanup), and
* the import-failure fallbacks of ``export_merged()`` / ``export_onnx()``.

No network, GPU, model download, Neo4j or large files are touched. Heavy
third-party imports and subprocess/model calls are replaced with in-memory
fakes. The real model-loading *happy* paths are deliberately not covered here:
they require downloaded weights and hardware this suite must never touch.
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import asdict, is_dataclass

import pytest

from src import model_export as ME

# Keep references to the real functions before any autouse patching.
_REAL_MERGED = ME.export_merged
_REAL_ONNX = ME.export_onnx


class _FakeOs:
    """Hermetic stand-in for the module's ``os`` (process-global os untouched)."""

    def __init__(self, exists_fn):
        self.path = types.SimpleNamespace(
            exists=exists_fn,
            dirname=os.path.dirname,
            expanduser=os.path.expanduser,
            join=os.path.join,
            getsize=os.path.getsize,
        )
        self.makedirs_calls = []

    def makedirs(self, path, exist_ok=False):
        self.makedirs_calls.append((path, exist_ok))


@pytest.fixture(autouse=True)
def forbid_real_model_loads(monkeypatch):
    """Never allow the model-loading functions to run in this offline suite."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a real model export was reached")

    monkeypatch.setattr(ME, "export_merged", forbidden)
    monkeypatch.setattr(ME, "export_onnx", forbidden)


def _stub(result):
    """Return (fake export fn, call log) for main() dispatch assertions."""
    calls = []

    def fake(*args):
        calls.append(args)
        return result

    return fake, calls


# --------------------------------------------------------------------------- #
# ExportResult (pure dataclass)
# --------------------------------------------------------------------------- #

def test_export_result_defaults():
    r = ME.ExportResult(format="merged", output_path="/tmp/out")
    assert r.format == "merged"
    assert r.output_path == "/tmp/out"
    assert r.size_bytes == 0
    assert r.success is False
    assert r.error is None
    assert r.duration_seconds == 0.0


def test_export_result_is_dataclass_with_stable_asdict():
    r = ME.ExportResult(format="gguf-q8_0", output_path="out.gguf",
                        size_bytes=12, success=True, error=None,
                        duration_seconds=1.5)
    assert is_dataclass(r)
    assert asdict(r) == {
        "format": "gguf-q8_0",
        "output_path": "out.gguf",
        "size_bytes": 12,
        "success": True,
        "error": None,
        "duration_seconds": 1.5,
    }


def test_export_result_equality_and_mutation():
    a = ME.ExportResult("onnx", "o")
    b = ME.ExportResult("onnx", "o")
    assert a == b
    b.error = "boom"
    assert a != b
    assert a.error is None


# --------------------------------------------------------------------------- #
# main() -- argparse + dispatch (no real export)
# --------------------------------------------------------------------------- #

def test_main_dispatches_merged_by_default(monkeypatch, capsys):
    ok = ME.ExportResult("merged", "out", success=True,
                         size_bytes=1_200_000, duration_seconds=2.25)
    fake, calls = _stub(ok)
    monkeypatch.setattr(ME, "export_merged", fake)

    rc = ME.main(["--checkpoint", "ckpt", "--output", "out"])

    assert rc == 0
    assert calls == [("ckpt", "out", "Qwen/Qwen2.5-0.5B-Instruct")]
    printed = capsys.readouterr().out
    assert "[export] Success: merged -> out" in printed
    assert "1.2MB" in printed
    assert "2.2s" in printed  # f"{2.25:.1f}" uses round-half-to-even


def test_main_dispatches_gguf_with_quantize(monkeypatch, capsys):
    ok = ME.ExportResult("gguf-q4_k_m", "out.gguf", success=True)
    fake, calls = _stub(ok)
    monkeypatch.setattr(ME, "export_gguf", fake)

    rc = ME.main(["--checkpoint", "ckpt", "--output", "out.gguf",
                  "--format", "gguf", "--quantize", "q4_k_m"])

    assert rc == 0
    assert calls == [("ckpt", "out.gguf", "q4_k_m")]
    assert "[export] Success: gguf-q4_k_m -> out.gguf" in capsys.readouterr().out


def test_main_dispatches_onnx_with_custom_base_model(monkeypatch):
    ok = ME.ExportResult("onnx", "onnxdir", success=True)
    fake, calls = _stub(ok)
    monkeypatch.setattr(ME, "export_onnx", fake)

    rc = ME.main(["--checkpoint", "ckpt", "--output", "onnxdir",
                  "--format", "onnx", "--base-model", "local/LFM3-1.7B"])

    assert rc == 0
    assert calls == [("ckpt", "onnxdir", "local/LFM3-1.7B")]


def test_main_returns_one_and_prints_failure(monkeypatch, capsys):
    bad = ME.ExportResult("merged", "out", success=False, error="merge exploded")
    fake, calls = _stub(bad)
    monkeypatch.setattr(ME, "export_merged", fake)

    rc = ME.main(["--checkpoint", "ckpt", "--output", "out"])

    assert rc == 1
    assert calls == [("ckpt", "out", "Qwen/Qwen2.5-0.5B-Instruct")]
    assert "[export] Failed: merge exploded" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--output", "out"],                                   # no checkpoint
    ["--checkpoint", "ckpt"],                              # no output
    ["--checkpoint", "c", "--output", "o", "--format", "bogus"],  # bad choice
])
def test_main_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        ME.main(argv)


# --------------------------------------------------------------------------- #
# export_gguf() -- early/offline failure and success branches
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("quantize", ["q8_0", "q4_k_m", ""])
def test_export_gguf_missing_converter_is_offline_failure(monkeypatch, quantize):
    fake_os = _FakeOs(lambda path: False)
    monkeypatch.setattr(ME, "os", fake_os)

    r = ME.export_gguf("ckpt", "out.gguf", quantize=quantize)

    assert r.format == f"gguf-{quantize}"
    assert r.output_path == "out.gguf"
    assert r.success is False
    assert r.error == "llama.cpp convert script not found"
    assert r.size_bytes == 0
    assert r.duration_seconds >= 0.0
    assert fake_os.makedirs_calls == []  # returned before any directory work


def test_export_gguf_reports_merge_failure(monkeypatch):
    output = "sub/out.gguf"
    fake_os = _FakeOs(
        lambda path: str(path).endswith("convert_hf_to_gguf.py"))
    monkeypatch.setattr(ME, "os", fake_os)
    monkeypatch.setattr(
        ME, "export_merged",
        lambda *a: ME.ExportResult("merged", output + ".merged",
                                   error="adapter boom"))

    r = ME.export_gguf("ckpt", output)

    assert r.success is False
    assert r.error == "Merge failed: adapter boom"
    assert r.format == "gguf-q8_0"
    assert fake_os.makedirs_calls == [("sub", True)]


def test_export_gguf_propagates_converter_stderr(monkeypatch):
    output = "sub/out.gguf"
    merged = output + ".merged"
    fake_os = _FakeOs(
        lambda path: str(path).endswith("convert_hf_to_gguf.py"))
    monkeypatch.setattr(ME, "os", fake_os)
    monkeypatch.setattr(
        ME, "export_merged",
        lambda *a: ME.ExportResult("merged", merged, success=True))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return types.SimpleNamespace(returncode=2, stderr="boom" * 200)

    monkeypatch.setattr(ME, "subprocess", types.SimpleNamespace(run=fake_run))

    r = ME.export_gguf("ckpt", output, quantize="q5_k_m")

    assert r.success is False
    assert r.error == ("boom" * 200)[-500:]  # stderr is truncated to 500 chars
    assert r.format == "gguf-q5_k_m"
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("convert_hf_to_gguf.py")
    assert cmd[2] == merged
    assert cmd[3:5] == ["--outfile", output]
    assert cmd[5:7] == ["--outtype", "q5_k_m"]
    assert kwargs == {"capture_output": True, "text": True, "timeout": 600}


def test_export_gguf_success_records_size_and_cleans_merged(monkeypatch, tmp_path):
    output = str(tmp_path / "out.gguf")
    merged = output + ".merged"
    os.mkdir(merged)                       # tiny real dir under pytest tmp_path
    (tmp_path / "out.gguf").write_bytes(b"GGUF" * 16)  # 64 bytes

    def exists(path):
        p = str(path)
        return (p.endswith("convert_hf_to_gguf.py")
                or p == output or p == merged)

    monkeypatch.setattr(ME, "os", _FakeOs(exists))
    monkeypatch.setattr(
        ME, "export_merged",
        lambda *a: ME.ExportResult("merged", merged, success=True))

    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(ME, "subprocess", types.SimpleNamespace(run=fake_run))

    r = ME.export_gguf("ckpt", output)

    assert r.success is True
    assert r.error is None
    assert r.size_bytes == 64
    assert r.output_path == output
    assert not os.path.exists(merged)      # shutil.rmtree cleaned it up


def test_export_gguf_unexpected_error_is_captured(monkeypatch):
    fake_os = _FakeOs(
        lambda path: str(path).endswith("convert_hf_to_gguf.py"))

    def refuse_makedirs(path, exist_ok=False):
        raise RuntimeError("makedirs refused")

    fake_os.makedirs = refuse_makedirs
    monkeypatch.setattr(ME, "os", fake_os)

    r = ME.export_gguf("ckpt", "sub/out.gguf")

    assert r.success is False
    assert r.error == "makedirs refused"
    assert r.format == "gguf-q8_0"


# --------------------------------------------------------------------------- #
# export_merged()/export_onnx() -- deterministic import-failure fallback
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("real_fn,expected_format", [
    (_REAL_MERGED, "merged"),
    (_REAL_ONNX, "onnx"),
])
def test_heavy_import_failure_is_captured(monkeypatch, real_fn, expected_format):
    # `None` in sys.modules makes import/from-import raise ImportError without
    # reaching any model, GPU or network.
    for name in ("transformers", "peft", "torch"):
        monkeypatch.setitem(sys.modules, name, None)

    r = real_fn("ckpt", "out")

    assert r.format == expected_format
    assert r.output_path == "out"
    assert r.success is False
    assert r.size_bytes == 0
    assert r.error and any(
        name in r.error for name in ("transformers", "peft", "torch"))
