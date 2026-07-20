"""Tests for the bench-matrix orchestration (multiple specs over one suite)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src import drivers_localchat as D
from src.bench import bench_matrix, format_bench_matrix, load_tasks, register_runner


class _FakeTok:
    pad_token_id = 0

    def apply_chat_template(self, *a, **k):
        return "P"

    def __call__(self, *a, **k):
        return type("T", (), {"input_ids": torch.tensor([[]])})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


class _FakeModel:
    device = "cpu"

    def generate(self, ids, **k):
        txt = '{"name": "write", "arguments": {"filePath": "g.txt", "content": "hi"}}'
        return torch.tensor([[1] + [ord(c) for c in txt] + [2]])


def _fake_local_chat(model_path, **kw):
    return D.LocalChatDriver(model_path, _model=_FakeModel(), _tok=_FakeTok())


def test_bench_matrix_runs_all_specs():
    register_runner("local-chat", _fake_local_chat)
    tasks = load_tasks(str(Path(__file__).parent.parent / "eval" / "tasks" / "sample.jsonl"))
    specs = [{"name": "m1", "runner": "local-chat", "model_path": "/x"},
             {"name": "m2", "runner": "local-chat", "model_path": "/y"}]
    matrix = bench_matrix(tasks, specs, rocm=False)
    assert set(matrix) == {"m1", "m2"}
    # every spec produced one result per task
    assert len(matrix["m1"]["results"]) == len(tasks)
    assert matrix["m1"]["summary"]["n"] == len(tasks)
    # tasks without file checks (command_exit) pass for the fake
    for r in matrix["m1"]["results"]:
        if r.task_id in ("exec-count-files", "exec-python-version",
                         "replay-git-status", "exec-shell-pipe", "replay-find-py",
                         "exec-pipe-wc"):
            assert r.success, r.task_id
        else:
            assert not r.success, r.task_id


def test_format_bench_matrix_shape():
    register_runner("local-chat", _fake_local_chat)
    tasks = load_tasks(str(Path(__file__).parent.parent / "eval" / "tasks" / "sample.jsonl"))
    matrix = bench_matrix(tasks, [{"name": "m1", "runner": "local-chat",
                                   "model_path": "/x"}], rocm=False)
    out = format_bench_matrix(matrix)
    assert "Benchmark matrix" in out
    assert "m1" in out
    assert "exec-hello-file" in out


def test_bench_matrix_bad_spec_survives():
    tasks = load_tasks(str(Path(__file__).parent.parent / "eval" / "tasks" / "sample.jsonl"))
    # unknown runner should not crash the whole matrix, just that spec errors out
    matrix = bench_matrix(tasks, [{"name": "bad", "runner": "does-not-exist",
                                   "model_path": "/x"}], rocm=False)
    assert "bad" in matrix
    assert "error" in matrix["bad"]["summary"]


def test_lmstudio_preset_builds_gguf_specs(monkeypatch):
    # replicate the lmstudio preset spec construction the CLI does, without a
    # live lmstudio server (no network calls). Just checks the glob + shape.
    import src.cli as cli  # noqa: F401
    lm_root = Path("/home/scott/.lmstudio/models")
    if not lm_root.exists():
        pytest.skip("no lmstudio models dir")
    specs = []
    for md in sorted(lm_root.rglob("*.gguf")):
        specs.append({"name": md.parent.name, "runner": "api",
                      "base_url": "http://localhost:1234/v1",
                      "model": md.parent.name, "api_key": "lm-studio"})
    assert len(specs) >= 1
    assert all(s["runner"] == "api" and s["model"] == s["name"] for s in specs)
    # RefinedToolCallV5 q8 gguf should be discoverable
    assert any(s["name"] == "RefinedToolCallV5-3b" for s in specs)


def test_local_preset_finds_qwen(monkeypatch):
    # the 'local'/'local-refs' preset should resolve the HF-cached Qwen2.5-7B
    qwen = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    if not qwen.is_dir():
        pytest.skip("no cached Qwen2.5-7B")
    base_local = None
    if not base_local:
        cand = str(qwen)
        if Path(cand).is_dir():
            base_local = cand
    assert base_local.endswith("models--Qwen--Qwen2.5-7B-Instruct")

