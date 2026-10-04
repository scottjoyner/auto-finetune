"""Tests for post-merge quantization.

Quantization is easy to get quietly wrong: a bad calibration set still produces
a loadable model that benchmarks worse than the original, and a failed run can
leave a partial directory that status reporting then treats as valid. These
tests pin the argument handling, calibration selection and output handling
without loading a model or needing auto-gptq/autoawq installed.
"""
from __future__ import annotations

import json
import os

import pytest

# Resolve transformers' lazy auto-class chain at collection time. Reaching
# transformers.AutoTokenizer lazily *during* a test re-enters the module
# __getattr__ and trips a circular import (auto_factory -> generation) that only
# manifests once another module has started importing transformers.
import transformers  # noqa: F401
from conftest import make_cfg
from transformers import AutoTokenizer  # noqa: F401

from src.quantize import (
    BUILTIN_CALIBRATION_PROMPTS,
    _get_dir_size_mb,
    _row_text,
    calibration_from_corpus,
    main,
    quantize_gptq,
)


def _merged(tmp_path, label: str = "combined") -> str:
    out_base = os.path.join(str(tmp_path), "checkpoints")
    src = os.path.join(out_base, f"toolcall-v5-3b-{label}-merged")
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "config.json"), "w") as fh:
        fh.write("{}")
    return src


# --- row flattening --------------------------------------------------------

def test_row_text_from_messages():
    row = {"messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"}]}
    assert _row_text(row) == "hi\nhello"


def test_row_text_from_conversations():
    row = {"conversations": [{"from": "human", "value": "q"},
                             {"from": "gpt", "value": "a"}]}
    assert _row_text(row) == "q\na"


def test_row_text_skips_empty_content():
    row = {"messages": [{"role": "user", "content": "  "},
                        {"role": "assistant", "content": "real"}]}
    assert _row_text(row) == "real"


def test_row_text_ignores_unknown_shapes():
    assert _row_text({"nope": 1}) == ""
    assert _row_text(["not", "a", "dict"]) == ""
    assert _row_text("string") == ""


# --- calibration selection -------------------------------------------------

def test_calibration_reads_corpus_and_honours_limit(tmp_path):
    ds = tmp_path / "datasets"
    ds.mkdir()
    rows = [{"messages": [{"role": "user", "content": f"sample-{i}"}]}
            for i in range(10)]
    (ds / "train.combined.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    texts = calibration_from_corpus(str(ds), "combined", limit=4)
    assert texts == ["sample-0", "sample-1", "sample-2", "sample-3"]


def test_calibration_skips_blank_and_corrupt_lines(tmp_path):
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "train.combined.jsonl").write_text(
        '{"messages":[{"role":"user","content":"good1"}]}\n'
        "\n"
        "{not json at all\n"
        '{"messages":[{"role":"user","content":"good2"}]}\n')
    assert calibration_from_corpus(str(ds), "combined", 10) == ["good1", "good2"]


def test_calibration_missing_corpus_returns_empty(tmp_path):
    # Empty means "fall back", not "quantize with nothing".
    assert calibration_from_corpus(str(tmp_path), "combined", 10) == []


def test_builtin_prompts_exist_but_are_small():
    # Guards the intent: this set is deliberately a reduced fallback, and the
    # quantizer must label it as such rather than presenting it as equivalent.
    assert len(BUILTIN_CALIBRATION_PROMPTS) >= 3


# --- quantize_gptq behaviour without a model ------------------------------

def _install_fake_gptq(monkeypatch, model):
    """Register a usable stand-in for auto_gptq.

    transformers probes availability with importlib.util.find_spec, which
    raises ValueError on a module object whose __spec__ was never set, so the
    fake needs a real ModuleSpec, not just a bare object in sys.modules.
    """
    import sys
    import types
    from importlib.machinery import ModuleSpec

    # Force transformers to finish importing FIRST. It caches
    # _auto_gptq_available at import time; if the fake is in sys.modules before
    # that probe runs, transformers believes auto_gptq is really installed and
    # walks an auto-modeling chain that is broken in this ROCm environment.

    mod = types.ModuleType("auto_gptq")
    mod.__spec__ = ModuleSpec("auto_gptq", None)
    mod.AutoGPTQForCausalLM = type("AutoGPTQForCausalLM", (), {
        "from_pretrained": staticmethod(lambda *a, **k: model)})
    mod.BaseQuantizeConfig = type("BaseQuantizeConfig", (), {
        "__init__": lambda self, **k: None})
    monkeypatch.setitem(sys.modules, "auto_gptq", mod)


class _FakeTok:
    def __call__(self, text, return_tensors=None):
        self.last = text

        class _T:
            input_ids = [1, 2, 3]
        return _T()

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


class _RecordingModel:
    """Records the calibration samples it was handed."""

    def __init__(self, payload=b"x" * 1024):
        self.texts: list = []
        self.payload = payload

    def quantize(self, cal_data, batch_size=1):
        self.texts = list(cal_data)

    def save_quantized(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "q.bin"), "wb") as fh:
            fh.write(self.payload)


def test_gptq_reports_reduced_calibration_when_none_supplied(tmp_path, monkeypatch):
    src = _merged(tmp_path)
    out = os.path.join(str(tmp_path), "out")
    model = _RecordingModel()
    _install_fake_gptq(monkeypatch, model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeTok()))

    result = quantize_gptq(src, out, bits=4)
    assert result.success, result.message
    assert len(model.texts) == len(BUILTIN_CALIBRATION_PROMPTS)
    assert "WARNING" in result.message
    assert "reduced" in result.message


def test_gptq_uses_supplied_corpus_texts_and_reports_count(tmp_path, monkeypatch):
    src = _merged(tmp_path)
    out = os.path.join(str(tmp_path), "out")
    tok, model = _FakeTok(), _RecordingModel(payload=b"y" * 2048)
    _install_fake_gptq(monkeypatch, model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: tok))

    corpus = [f"real-sample-{i}" for i in range(6)]
    result = quantize_gptq(src, out, bits=4, dataset_size=3,
                           calibration_texts=corpus)
    assert result.success, result.message
    # dataset_size must actually bound the calibration set now.
    assert len(model.texts) == 3
    assert "3 corpus calibration samples" in result.message
    assert "WARNING" not in result.message


def test_gptq_clears_stale_output_dir(tmp_path, monkeypatch):
    src = _merged(tmp_path)
    out = os.path.join(str(tmp_path), "out")
    os.makedirs(out, exist_ok=True)
    stale = os.path.join(out, "stale-leftover.safetensors")
    with open(stale, "wb") as fh:
        fh.write(b"old")

    model = _RecordingModel(payload=b"new")
    _install_fake_gptq(monkeypatch, model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeTok()))

    result = quantize_gptq(src, out, bits=4, calibration_texts=["x"])
    assert result.success, result.message
    assert not os.path.exists(stale), "stale artifact survived quantization"
    assert os.path.exists(os.path.join(out, "q.bin"))


def test_gptq_missing_dependency_is_reported_not_raised(tmp_path):
    src = _merged(tmp_path)
    result = quantize_gptq(src, os.path.join(str(tmp_path), "out"))
    # auto_gptq is not installed here, so this must be a clean failure value.
    assert result.success is False
    assert result.output_path == ""
    assert "missing dependency" in result.message or "failed" in result.message


# --- CLI argument handling -------------------------------------------------

@pytest.mark.parametrize("argv,expect", [
    (["cli", "quantize", "--label=x", "--bits=abc"], 2),
    (["cli", "quantize", "--label=x", "--bits=7"], 2),
    (["cli", "quantize", "--label=x", "--bits=0"], 2),
    (["cli", "quantize", "--method=bogus"], 2),
    (["cli", "quantize", "--label=x", "--calibration-size=0"], 2),
    (["cli", "quantize", "--label=x", "--calibration-size=abc"], 2),
    (["cli", "quantize"], 2),
])
def test_cli_rejects_bad_arguments_without_crashing(argv, expect, capsys):
    # Every one of these used to either raise ValueError or silently proceed.
    assert main(make_cfg(), argv) == expect
    assert "[error]" in capsys.readouterr().out


def test_cli_missing_merged_model_is_reported(tmp_path, capsys):
    cfg = make_cfg(train={"output_dir": str(tmp_path / "checkpoints")})
    assert main(cfg, ["cli", "quantize", "--label=nope"]) == 2
    assert "merged model not found" in capsys.readouterr().out


def test_cli_passes_corpus_calibration_to_gptq(tmp_path, monkeypatch):
    _merged(tmp_path, "combined")
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "train.combined.jsonl").write_text(
        "\n".join(json.dumps({"messages": [{"role": "user", "content": f"c{i}"}]})
                  for i in range(5)))

    seen = {}

    def fake_gptq(model_path, output_path, bits=4, dataset_size=128,
                  calibration_texts=None):
        seen["texts"] = calibration_texts
        seen["dataset_size"] = dataset_size
        from src.quantize import QuantizeResult
        return QuantizeResult(True, "combined", model_path, output_path, bits,
                              100.0, 30.0, 3.3, 1.0, "ok")

    monkeypatch.setattr("src.quantize.quantize_gptq", fake_gptq)
    cfg = make_cfg(train={"output_dir": os.path.join(str(tmp_path), "checkpoints")},
                   paths={"dataset_dir": str(ds)})
    rc = main(cfg, ["cli", "quantize", "--label=combined",
                    "--calibration-size=3"])
    assert rc == 0
    assert seen["texts"] == ["c0", "c1", "c2"]
    assert seen["dataset_size"] == 3


# --- helpers ---------------------------------------------------------------

def test_get_dir_size_mb(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * (1024 * 1024))
    assert _get_dir_size_mb(str(tmp_path)) == pytest.approx(2.0)


def test_quantize_status_lists_quantized_dirs(tmp_path, capsys):
    out_base = tmp_path / "checkpoints"
    (out_base / "toolcall-v5-3b-combined-merged").mkdir(parents=True)
    (out_base / "toolcall-v5-3b-combined-merged-4bit").mkdir(parents=True)
    with open(out_base / "toolcall-v5-3b-combined-merged-4bit" / "q.bin", "wb") as fh:
        fh.write(b"z" * 1024)

    cfg = make_cfg(train={"output_dir": str(out_base)})
    assert main(cfg, ["cli", "quantize-status"]) == 0
    out = capsys.readouterr().out
    assert "merged-4bit" in out
    # the un-quantized merged dir must not be reported as quantized
    assert out.count("toolcall-v5-3b-") == 1
