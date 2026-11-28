"""Configuration validation safety tests using only synthetic, temporary paths."""
from __future__ import annotations


import pytest

from src import config_validator as validator


class StubConfig:
    def __init__(self, root):
        self.paths = {
            name: str(root / name)
            for name in ("raw_dir", "cleaned_dir", "dataset_dir", "analysis_dir")
        }
        self.sections = {
            "train": {"backend": "auto", "model_name": "LiquidAI/LFM3-1.7B",
                      "max_seq_length": 2048},
            "sources": {"opencode": {"enabled": False}, "hermes": {"enabled": False}},
            "extract": {"min_messages": 0},
            "clean": {"max_chars_per_message": 2000},
            "format": {"template": "chatml"},
        }

    def path(self, name):
        return self.paths.get(name)

    def get(self, key, default=None):
        return self.sections.get(key, default)


@pytest.fixture
def config(tmp_path):
    return StubConfig(tmp_path)


def test_valid_synthetic_configuration_accepted(config):
    assert validator.validate(config) == []


@pytest.mark.parametrize("name", validator.REQUIRED_PATHS)
def test_missing_required_path_reported(config, name):
    config.paths[name] = None
    assert f"Missing path: {name}" in validator.validate(config)


@pytest.mark.parametrize("name", validator.REQUIRED_PATHS)
def test_relative_required_path_reported(config, name):
    config.paths[name] = "relative/" + name
    assert any(f"Path not absolute: {name}" in x for x in validator.validate(config))


def test_missing_path_method_not_considered_valid(config):
    class NoPaths:
        def get(self, key, default=None):
            return config.get(key, default)
    result = validator.validate(NoPaths())
    assert sum("Missing path:" in s for s in result) == len(validator.REQUIRED_PATHS)


@pytest.mark.parametrize("backend", ["unknown", "", "ROCm", None])
def test_invalid_training_backend_denied(config, backend):
    config.sections["train"]["backend"] = backend
    assert any("Invalid backend:" in x for x in validator.validate(config))


@pytest.mark.parametrize("model", ["", None])
def test_missing_model_denied(config, model):
    config.sections["train"]["model_name"] = model
    assert "Missing train.model_name" in validator.validate(config)


@pytest.mark.parametrize("length", [0, -1, None, 1.5, "8192"])
def test_invalid_max_sequence_length_denied(config, length):
    config.sections["train"]["max_seq_length"] = length
    assert any("Invalid max_seq_length:" in x for x in validator.validate(config))


def test_empty_sources_denied(config):
    config.sections["sources"] = {}
    assert "No sources configured" in validator.validate(config)


@pytest.mark.parametrize(("source", "key"), [
    ("opencode", "db_path"),
    ("hermes", "state_db"),
])
def test_missing_enabled_source_db_denied(config, source, key, tmp_path):
    missing = str(tmp_path / "missing.db")
    config.sections["sources"][source] = {"enabled": True, key: missing}
    assert any(missing in x and "not found" in x for x in validator.validate(config))


@pytest.mark.parametrize(("source", "key"), [
    ("opencode", "db_path"),
    ("hermes", "state_db"),
])
def test_existing_enabled_source_db_accepted(config, source, key, tmp_path):
    path = tmp_path / "synthetic.db"
    path.write_bytes(b"not a real DB; only file existence is validated here")
    config.sections["sources"][source] = {"enabled": True, key: str(path)}
    assert validator.validate(config) == []


@pytest.mark.parametrize(("section", "key", "values", "message"), [
    ("extract", "min_messages", [-1, "many", None], "Invalid min_messages"),
    ("clean", "max_chars_per_message", [0, -1, "x"], "Invalid max_chars_per_message"),
    ("format", "template", ["unknown", None, ""], "Invalid format template"),
])
def test_bad_processing_settings_denied(config, section, key, values, message):
    for value in values:
        config.sections[section][key] = value
        assert any(message in e for e in validator.validate(config))


def test_report_good_and_bad_configurations(config, capsys):
    assert validator.validate_and_report(config) is True
    assert "Configuration valid" in capsys.readouterr().out
    config.sections["train"]["model_name"] = ""
    assert validator.validate_and_report(config) is False
    assert "Missing train.model_name" in capsys.readouterr().out


@pytest.mark.parametrize(("strict", "expected"), [(True, 1), (False, 0)])
def test_cli_reports_issues_with_strict_exit(config, monkeypatch, capsys, strict, expected):
    config.sections["clean"]["max_chars_per_message"] = 0
    monkeypatch.setattr(validator, "load", lambda: config)
    assert validator.main(["--strict"] if strict else []) == expected
    assert "Invalid max_chars_per_message" in capsys.readouterr().out


def test_cli_accepts_healthy_config(config, monkeypatch, capsys):
    monkeypatch.setattr(validator, "load", lambda: config)
    assert validator.main(["--strict"]) == 0
    assert "Configuration valid" in capsys.readouterr().out
