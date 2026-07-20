"""Tests for src.cli dispatch."""
from __future__ import annotations

import sys
import copy
import types
from pathlib import Path

import pytest

from src.cli import main as cli_main
from src.train import TrainError


def test_cli_help():
    assert cli_main(["cli", "help"]) == 0
    assert cli_main(["cli"]) == 0


def test_cli_unknown_command():
    # unknown command falls through to help
    assert cli_main(["cli", "bogus"]) == 0


def test_cli_extract_runs(make_opencode_db, tmp_path, monkeypatch):
    raw = copy.deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
    raw["sources"]["opencode"]["db_path"] = make_opencode_db()
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    import src.cli as cli
    cfg = __import__("src.config", fromlist=["Config"]).Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    rc = cli_main(["cli", "extract"])
    assert rc == 1
    assert (tmp_path / "raw" / "ses_demo.json").exists()


def test_cli_train_error_propagates(tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    ds.mkdir()
    raw = copy.deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(ds)}
    import src.cli as cli
    cfg = __import__("src.config", fromlist=["Config"]).Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    # no train.jsonl -> TrainError -> exit code 2
    rc = cli_main(["cli", "train"])
    assert rc == 2


def test_cli_train_dry_run(tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "train.jsonl").write_text('{"messages":[{"role":"user","content":"q"}]}\n')
    raw = copy.deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(ds)}
    import src.cli as cli
    cfg = __import__("src.config", fromlist=["Config"]).Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)

    # stub transformers.AutoTokenizer used by dry-run
    fake_tf = types.ModuleType("transformers")
    class FakeTok:
        @staticmethod
        def from_pretrained(name): return FakeTok()
        def apply_chat_template(self, *a, **k): return "x"
    fake_tf.AutoTokenizer = FakeTok
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_tf
    try:
        rc = cli_main(["cli", "train", "--dry-run"])
    finally:
        if saved: sys.modules["transformers"] = saved
        else: sys.modules.pop("transformers", None)
    assert rc == 0


def test_cli_train_passes_label(monkeypatch):
    import src.cli as cli
    from src.config import Config, _DEFAULTS

    raw = copy.deepcopy(_DEFAULTS)
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)

    captured = {}

    def fake_train(cfg, dry_run=False, source=None, label=None, max_examples=None):
        captured["dry_run"] = dry_run
        captured["source"] = source
        captured["label"] = label
        captured["max_examples"] = max_examples
        return 0

    monkeypatch.setattr("src.train.main", fake_train)

    cli_cli = cli_main(["cli", "train", "--label=ssd", "--source=opencode"])
    assert cli_cli == 0
    assert captured == {"dry_run": False, "source": "opencode", "label": "ssd", "max_examples": None}


def test_cli_clean_and_format(make_opencode_db, tmp_path, monkeypatch):
    import src.cli as cli
    from src.config import Config, _DEFAULTS
    raw = copy.deepcopy(_DEFAULTS)
    raw["sources"]["opencode"]["db_path"] = make_opencode_db()
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    assert cli_main(["cli", "extract"]) == 1
    assert isinstance(cli_main(["cli", "clean"]), int)
    assert isinstance(cli_main(["cli", "format"]), int)


def test_cli_hermes_disabled(make_opencode_db, tmp_path, monkeypatch):
    import src.cli as cli
    from src.config import Config, _DEFAULTS
    raw = copy.deepcopy(_DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    assert cli_main(["cli", "hermes"]) == 0


def test_cli_all(make_opencode_db, tmp_path, monkeypatch):
    import src.cli as cli
    from src.config import Config, _DEFAULTS
    raw = copy.deepcopy(_DEFAULTS)
    raw["sources"]["opencode"]["db_path"] = make_opencode_db()
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    assert cli_main(["cli", "all"]) == 0
    assert (tmp_path / "datasets" / "train.jsonl").exists()


def test_cli_eval_split_writes_heldout(tmp_path, monkeypatch):
    # eval-split calls build_held_out on the dataset dir: pure data, no model.
    import src.cli as cli
    from src.config import Config, _DEFAULTS
    import src.eval as E
    ds = tmp_path / "datasets"
    ds.mkdir()
    ds.joinpath("train.jsonl").write_text(
        "\n".join(f'{{"messages":[{{"role":"user","content":"q{i}"}}]}}' for i in range(10)))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    def fake_build_held_out(dset, label, frac=0.1):
        # mirror the CLI's output location
        out = Path(dset).parent / "eval" / f"held-out-{label}.jsonl"
        out.write_text("x")
        return out

    monkeypatch.setattr(E, "build_held_out", fake_build_held_out)
    raw = copy.deepcopy(_DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(ds)}
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    rc = cli_main(["cli", "eval-split", "--label=ssd", "--frac=0.2"])
    assert rc == 0
    assert (eval_dir / "held-out-ssd.jsonl").exists()


def _fake_driver_class():
    import importlib
    B = importlib.import_module("src.bench")

    class FakeDriver(B.ModelDriver):
        def __init__(self): self.n = 0
        def generate(self, messages, max_new_tokens=512):
            self.n += 1
            if self.n == 1:
                return ('<tool_call name="write" call_id="1">'
                        '{"filePath":"g.txt","content":"Hello, agent."}'
                        '\u276E\u276E\u276E')
            return "done"
    return B, FakeDriver


def test_cli_bench_runs_with_fake_driver(monkeypatch):
    # bench --runner=self should drive a fake driver end-to-end (no model).
    import importlib
    cli = importlib.import_module("src.cli")
    B, FakeDriver = _fake_driver_class()
    tasks = [B.Task.from_dict({
        "id": "t1", "prompt": "x",
        "checks": [{"kind": "file_exists", "path": "g.txt"},
                   {"kind": "file_contains", "path": "g.txt", "expect": "Hello, agent."}]})]
    monkeypatch.setattr(B, "load_tasks", lambda p: tasks)
    monkeypatch.setattr(B, "make_driver", lambda runner, **kw: FakeDriver())
    rc = cli.main(["cli", "bench", "--runner=self", "--model=/tmp/fake",
                   "--tasks=/tmp/none"])
    assert rc == 0


def test_cli_bench_matrix_runs_with_fake_specs(monkeypatch, capsys):
    # bench-matrix with explicit --specs using a fake runner: no model loads.
    import importlib
    cli = importlib.import_module("src.cli")
    B, FakeDriver = _fake_driver_class()
    tasks = [B.Task.from_dict({
        "id": "t1", "prompt": "x",
        "checks": [{"kind": "file_exists", "path": "g.txt"},
                   {"kind": "file_contains", "path": "g.txt", "expect": "hi"}]})]
    monkeypatch.setattr(B, "load_tasks", lambda p: tasks)
    monkeypatch.setattr(B, "make_driver", lambda runner, **kw: FakeDriver())
    specs = '[{"name":"fake1","runner":"self","model_path":"/x"}]'
    rc = cli.main(["cli", "bench-matrix", f"--specs={specs}", "--tasks=/tmp/none"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fake1" in out and "t1" in out


def test_cli_bench_matrix_rejects_no_args():
    # no --specs / --preset -> clean error, exit 2
    assert cli_main(["cli", "bench-matrix"]) == 2


def test_cli_bench_matrix_writes_report(monkeypatch):
    import importlib
    cli = importlib.import_module("src.cli")
    B, FakeDriver = _fake_driver_class()
    tasks = [B.Task.from_dict({
        "id": "t1", "prompt": "x",
        "checks": [{"kind": "file_exists", "path": "g.txt"}]})]
    monkeypatch.setattr(B, "load_tasks", lambda p: tasks)
    monkeypatch.setattr(B, "make_driver", lambda runner, **kw: FakeDriver())
    specs = '[{"name":"fake1","runner":"self","model_path":"/x"}]'
    rc = cli.main(["cli", "bench-matrix", f"--specs={specs}",
                   "--tasks=/tmp/none", "--report"])
    assert rc == 0


def test_cli_bench_matrix_all_preset_combines_sources(monkeypatch, capsys):
    # --preset=all must include all three reference sources: local-chat (qwen),
    # lmstudio gguf (api), and fleet (api). No model loads (faked).
    import importlib
    cli = importlib.import_module("src.cli")
    B = importlib.import_module("src.bench")

    captured = {}
    def fake_matrix(tasks, specs, rocm=False):
        captured["specs"] = specs
        return {sp["name"]: {"results": [], "summary": {"n": 0, "passed": 0,
                "completed": 0, "errors": 0, "pass_rate": 0.0}} for sp in specs}
    monkeypatch.setattr(B, "bench_matrix", fake_matrix)
    monkeypatch.setattr(B, "load_tasks", lambda p: [])
    rc = cli.main(["cli", "bench-matrix", "--preset=all", "--tasks=/tmp/none"])
    assert rc == 0
    specs = captured["specs"]
    runners = {s["runner"] for s in specs}
    assert "local-chat" in runners      # local Qwen2.5-7B reference
    assert "api" in runners             # lmstudio + fleet
    assert any(s["name"] == "qwen2.5-7b" for s in specs)


def test_cli_bench_matrix_local_ref_specs_helper(monkeypatch):
    import importlib
    cli = importlib.import_module("src.cli")
    specs = cli._local_ref_specs()
    # qwen2.5-7b local-chat reference should always be present (HF cache exists)
    assert any(s["name"] == "qwen2.5-7b" and s["runner"] == "local-chat"
               for s in specs)
    # any finished adapters on disk are appended as subagent specs
    for s in specs:
        if s["name"].startswith("ft-"):
            assert s["runner"] == "subagent"


def test_cli_bench_matrix_fleet_preset_no_crash(monkeypatch, capsys):
    # --preset=fleet builds api specs from endpoints.json; must not raise even
    # if individual nodes are down (per-spec failures handled at run time).
    import importlib
    cli = importlib.import_module("src.cli")
    B = importlib.import_module("src.bench")
    captured = {}
    def fake_matrix(tasks, specs, rocm=False):
        captured["specs"] = specs
        return {sp["name"]: {"results": [], "summary": {"n": 0, "passed": 0,
                "completed": 0, "errors": 0, "pass_rate": 0.0}} for sp in specs}
    monkeypatch.setattr(B, "bench_matrix", fake_matrix)
    monkeypatch.setattr(B, "load_tasks", lambda p: [])
    rc = cli.main(["cli", "bench-matrix", "--preset=fleet", "--tasks=/tmp/none"])
    assert rc == 0
    assert len(captured["specs"]) >= 1
    assert all(s["runner"] == "api" for s in captured["specs"])


def test_cli_bench_matrix_fast_preset_smoke(monkeypatch, capsys):
    # --preset=fast runs ONE model per source (local + lmstudio + fleet).
    import importlib
    cli = importlib.import_module("src.cli")
    B = importlib.import_module("src.bench")
    captured = {}
    def fake_matrix(tasks, specs, rocm=False):
        captured["specs"] = specs
        return {sp["name"]: {"results": [], "summary": {"n": 0, "passed": 0,
                "completed": 0, "errors": 0, "pass_rate": 0.0}} for sp in specs}
    monkeypatch.setattr(B, "bench_matrix", fake_matrix)
    monkeypatch.setattr(B, "load_tasks", lambda p: [])
    rc = cli.main(["cli", "bench-matrix", "--preset=fast", "--tasks=/tmp/none"])
    assert rc == 0
    specs = captured["specs"]
    # at least the local qwen reference, and no more than one per source family
    assert any(s["name"] == "qwen2.5-7b" for s in specs)
    assert len([s for s in specs if s["runner"] == "local-chat"]) <= 1
