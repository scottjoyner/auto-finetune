"""Tests for src.cli dispatch."""
from __future__ import annotations

import sys
import types

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
    raw = __import__("copy").deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
    raw["sources"]["opencode"]["db_path"] = make_opencode_db()
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    # write minimal config.yaml so load() picks it up via cwd? load uses project
    # root config.yaml; instead monkeypatch load to return our cfg.
    import src.cli as cli
    cfg = __import__("src.config", fromlist=["Config"]).Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    rc = cli_main(["cli", "extract"])
    assert rc == 1
    assert (tmp_path / "raw" / "ses_demo.json").exists()


def test_cli_train_error_propagates(tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    ds.mkdir()
    raw = __import__("copy").deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
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
    raw = __import__("copy").deepcopy(__import__("src.config", fromlist=["_DEFAULTS"])._DEFAULTS)
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(ds)}
    import src.cli as cli
    cfg = __import__("src.config", fromlist=["Config"]).Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)

    # stub transformers.AutoTokenizer used by dry-run
    import sys
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
    import copy

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
    import copy
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
    import copy
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
    import copy
    raw = copy.deepcopy(_DEFAULTS)
    raw["sources"]["opencode"]["db_path"] = make_opencode_db()
    raw["paths"] = {"raw_dir": str(tmp_path / "raw"),
                    "cleaned_dir": str(tmp_path / "cleaned"),
                    "dataset_dir": str(tmp_path / "datasets")}
    cfg = Config(raw=raw)
    monkeypatch.setattr(cli, "load", lambda *a, **k: cfg)
    assert cli_main(["cli", "all"]) == 0
    assert (tmp_path / "datasets" / "train.jsonl").exists()
