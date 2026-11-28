"""Tests for src.config."""
from __future__ import annotations

import importlib.util
import os

from src.config import Config, _deep_merge, load, project_root


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20, "z": 30}, "c": 4}
    merged = _deep_merge(base, over)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"y": 2}})
    assert base == {"a": {"x": 1}}


def test_load_reads_user_config(tmp_root):
    # write a user config overriding template
    (tmp_root / "config.yaml").write_text("format:\n  template: sharegpt\n")
    cfg = load(str(tmp_root / "config.yaml"))
    assert cfg.get("format", "template") == "sharegpt"
    # unspecified keys fall back to defaults
    assert cfg.get("train", "lora_r") == 32


def test_load_missing_file_uses_defaults():
    cfg = load("/nonexistent/path/config.yaml")
    assert cfg.get("format", "template") == "chatml"
    assert isinstance(cfg.raw, dict) and cfg.raw


def test_get_missing_returns_default():
    cfg = Config(raw={})
    assert cfg.get("nope", "key", default=42) == 42
    assert cfg.get("format", "missing", default="x") == "x"


def test_get_nested():
    cfg = Config(raw={"a": {"b": {"c": 7}}})
    assert cfg.get("a", "b", "c") == 7


def test_paths_are_absolute_or_relative_to_root():
    cfg = Config(raw={"paths": {"raw_dir": "data/raw"}})
    assert cfg.paths["raw_dir"].endswith(os.path.join("data", "raw"))
    # absolute path passes through unchanged
    cfg2 = Config(raw={"paths": {"raw_dir": "/abs/dir"}})
    assert cfg2.paths["raw_dir"] == "/abs/dir"


def test_path_helper_default():
    cfg = Config(raw={})
    assert cfg.path("raw_dir").endswith(os.path.join("data", "raw_dir"))


def test_ensure_dirs(tmp_path):
    cfg = Config(raw={"paths": {
        "raw_dir": str(tmp_path / "r"),
        "cleaned_dir": str(tmp_path / "c"),
        "dataset_dir": str(tmp_path / "d"),
    }})
    cfg.ensure_dirs()
    assert (tmp_path / "r").is_dir()
    assert (tmp_path / "c").is_dir()
    assert (tmp_path / "d").is_dir()


def test_project_root_is_parent_of_src():
    root = project_root()
    src_dir = os.path.join(root, "src")
    assert os.path.isdir(src_dir)
    assert os.path.isfile(os.path.join(src_dir, "__init__.py"))
    # A checkout may be a git worktree with an arbitrary directory name, so the
    # basename is not part of the contract. What matters is that `src` is
    # importable from this root and that `src.config` resolves to the file
    # under it. A wrong root (another checkout, or one level too high/low)
    # still fails here even though its basename is arbitrary too.
    spec = importlib.util.find_spec("src.config")
    assert spec is not None and spec.origin
    assert os.path.realpath(spec.origin) == os.path.realpath(
        os.path.join(src_dir, "config.py")
    )


from src.config import Config
from src.config_validator import validate, is_lfm3_config


def test_validate_lfm3_max_seq_length_issue():
    cfg = Config(raw={"train": {"model_name": "LiquidAI/LFM3-1.7B",
                                "max_seq_length": 8192,
                                "load_in_4bit": False}})
    issues = validate(cfg)
    assert any("max_seq_length" in i and "4096" in i for i in issues)


def test_validate_lfm3_load_in_4bit_warning():
    cfg = Config(raw={"train": {"model_name": "LiquidAI/LFM3-1.7B",
                                "max_seq_length": 2048,
                                "load_in_4bit": True}})
    issues = validate(cfg)
    assert any(i.startswith("Warning:") and "load_in_4bit" in i for i in issues)


def test_is_lfm3_config():
    cfg = Config(raw={"train": {"model_name": "LiquidAI/LFM3-1.7B"}})
    assert is_lfm3_config(cfg) is True
    cfg2 = Config(raw={"train": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}})
    assert is_lfm3_config(cfg2) is False
