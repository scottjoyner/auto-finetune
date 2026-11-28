"""Offline, deterministic coverage for src.config_validator's pure validation.

Everything here is backed by in-memory dicts: no config.yaml load, no network,
no GPU, no DB connections. Filesystem checks are limited to absolute path
strings and ``tmp_path`` (SSD scratch) probes for the two optional DB paths.
"""
from __future__ import annotations

import pytest

from src.config_validator import (
    REQUIRED_PATHS,
    VALID_BACKENDS,
    VALID_FORMATS,
    validate,
    validate_and_report,
)


class _DictCfg:
    """Dict-backed stand-in for Config.get(); deliberately has no .path()."""

    def __init__(self, raw=None):
        self.raw = dict(raw or {})

    def get(self, *keys, default=None):
        node = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


class _Cfg(_DictCfg):
    """As above, with a dict-backed .path() for the required-path checks."""

    def __init__(self, raw=None, paths=None):
        super().__init__(raw)
        self._paths = dict(paths or {})

    def path(self, name):
        return self._paths.get(name)


def _abs_paths(tmp_path):
    return {name: str(tmp_path / name) for name in REQUIRED_PATHS}


def _minimal_raw(**overrides):
    raw = {
        "train": {"backend": "auto", "model_name": "m", "max_seq_length": 8},
        "sources": {"opencode": {"enabled": False}, "hermes": {"enabled": False}},
        "extract": {"min_messages": 0},
        "clean": {"max_chars_per_message": 1},
        "format": {"template": "chatml"},
    }
    for key, value in overrides.items():
        raw[key] = value
    return raw


def _issues(cfg):
    return validate(cfg)


# --- required paths -------------------------------------------------------

def test_valid_config_has_no_issues(tmp_path):
    cfg = _Cfg(_minimal_raw(), _abs_paths(tmp_path))
    assert _issues(cfg) == []


def test_missing_path_attribute_reports_every_required_path():
    cfg = _DictCfg(_minimal_raw())
    assert _issues(cfg) == [f"Missing path: {p}" for p in REQUIRED_PATHS]


def test_missing_path_values_are_reported_in_order(tmp_path):
    paths = _abs_paths(tmp_path)
    paths.pop("cleaned_dir")
    paths.pop("dataset_dir")
    cfg = _Cfg(_minimal_raw(), paths)
    assert _issues(cfg) == [
        "Missing path: cleaned_dir",
        "Missing path: dataset_dir",
    ]


@pytest.mark.parametrize("bad", ["data/raw", ".", "", "~/data"])
def test_relative_paths_are_rejected(tmp_path, bad):
    paths = _abs_paths(tmp_path)
    paths["raw_dir"] = bad
    cfg = _Cfg(_minimal_raw(), paths)
    assert _issues(cfg) == [f"Path not absolute: raw_dir={bad}"]


def test_all_required_paths_absolute_and_existing(tmp_path):
    paths = {name: str(tmp_path / name) for name in REQUIRED_PATHS}
    for value in paths.values():
        (tmp_path / value).mkdir(parents=True, exist_ok=True)
    cfg = _Cfg(_minimal_raw(), paths)
    assert _issues(cfg) == []


# --- train section --------------------------------------------------------

@pytest.mark.parametrize("backend", sorted(VALID_BACKENDS))
def test_all_valid_backends_accepted(tmp_path, backend):
    raw = _minimal_raw(train={"backend": backend, "model_name": "m", "max_seq_length": 8})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


@pytest.mark.parametrize("backend", ["", None, "UNSLOTH", "transformers", 42])
def test_invalid_backend_reported(tmp_path, backend):
    raw = _minimal_raw(train={"backend": backend, "model_name": "m", "max_seq_length": 8})
    issues = _issues(_Cfg(raw, _abs_paths(tmp_path)))
    assert len(issues) == 1
    assert issues[0].startswith(f"Invalid backend: {backend} ")
    assert "must be one of" in issues[0]
    for name in VALID_BACKENDS:
        assert name in issues[0]


def test_missing_train_section_uses_defaults_and_reports_two_issues(tmp_path):
    raw = _minimal_raw(train=None)
    issues = _issues(_Cfg(raw, _abs_paths(tmp_path)))
    assert issues == ["Missing train.model_name", "Invalid max_seq_length: 0"]


@pytest.mark.parametrize("model_name", [None, "", 0])
def test_missing_model_name_reported(tmp_path, model_name):
    raw = _minimal_raw(train={"model_name": model_name, "max_seq_length": 8})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == ["Missing train.model_name"]


@pytest.mark.parametrize("bad", [None, 0, -1, "8192", 8192.0, [], {}])
def test_invalid_max_seq_length_reported(tmp_path, bad):
    raw = _minimal_raw(train={"backend": "auto", "model_name": "m", "max_seq_length": bad})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [f"Invalid max_seq_length: {bad}"]


def test_missing_max_seq_length_reported(tmp_path):
    raw = _minimal_raw(train={"backend": "auto", "model_name": "m"})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == ["Invalid max_seq_length: 0"]


def test_bool_true_max_seq_length_is_accepted_as_int(tmp_path):
    # isinstance(True, int) is True and True < 1 is False: document reality.
    raw = _minimal_raw(train={"backend": "auto", "model_name": "m", "max_seq_length": True})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


# --- sources / optional DB paths ------------------------------------------

@pytest.mark.parametrize("sources", [{}, None])
def test_no_sources_configured(tmp_path, sources):
    raw = _minimal_raw(sources=sources)
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == ["No sources configured"]


def test_sources_present_without_db_paths_is_valid(tmp_path):
    raw = _minimal_raw(sources={"opencode": {}, "hermes": {}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_missing_opencode_db_reported_when_enabled_by_default(tmp_path):
    db = str(tmp_path / "missing-opencode.db")
    raw = _minimal_raw(sources={"opencode": {"db_path": db}, "hermes": {"enabled": False}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [f"OpenCode DB not found: {db}"]


def test_existing_opencode_db_accepted(tmp_path):
    db = tmp_path / "opencode.db"
    db.write_text("")
    raw = _minimal_raw(sources={"opencode": {"db_path": str(db)}, "hermes": {"enabled": False}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_empty_opencode_db_path_is_ignored(tmp_path):
    raw = _minimal_raw(sources={"opencode": {"db_path": ""}, "hermes": {"enabled": False}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_disabled_opencode_skips_db_check(tmp_path):
    db = str(tmp_path / "missing-opencode.db")
    raw = _minimal_raw(sources={"opencode": {"enabled": False, "db_path": db},
                                "hermes": {"enabled": False}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_missing_hermes_state_db_reported_when_enabled_by_default(tmp_path):
    db = str(tmp_path / "missing-state.db")
    raw = _minimal_raw(sources={"opencode": {"enabled": False}, "hermes": {"state_db": db}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [f"Hermes state.db not found: {db}"]


def test_existing_hermes_state_db_accepted(tmp_path):
    db = tmp_path / "state.db"
    db.write_text("")
    raw = _minimal_raw(sources={"opencode": {"enabled": False}, "hermes": {"state_db": str(db)}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_disabled_hermes_skips_state_db_check(tmp_path):
    db = str(tmp_path / "missing-state.db")
    raw = _minimal_raw(sources={"opencode": {"enabled": False},
                                "hermes": {"enabled": False, "state_db": db}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_both_missing_dbs_reported_together(tmp_path):
    opencode_db = str(tmp_path / "no-opencode.db")
    hermes_db = str(tmp_path / "no-hermes.db")
    raw = _minimal_raw(sources={"opencode": {"db_path": opencode_db},
                                "hermes": {"state_db": hermes_db}})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [
        f"OpenCode DB not found: {opencode_db}",
        f"Hermes state.db not found: {hermes_db}",
    ]


# --- extract / clean / format ---------------------------------------------

@pytest.mark.parametrize("bad", [-1, -100, "2", 2.5, None])
def test_invalid_min_messages_reported(tmp_path, bad):
    raw = _minimal_raw(extract={"min_messages": bad})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [f"Invalid min_messages: {bad}"]


def test_missing_extract_section_is_valid(tmp_path):
    raw = _minimal_raw(extract=None)
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


@pytest.mark.parametrize("bad", [None, 0, -1, "32000", 32000.0])
def test_invalid_max_chars_reported(tmp_path, bad):
    raw = _minimal_raw(clean={"max_chars_per_message": bad})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [
        f"Invalid max_chars_per_message: {bad}"
    ]


def test_missing_clean_section_is_invalid(tmp_path):
    raw = _minimal_raw(clean=None)
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == ["Invalid max_chars_per_message: 0"]


@pytest.mark.parametrize("bad", ["", None, "CHATML", "bogus", 7])
def test_invalid_format_template_reported(tmp_path, bad):
    raw = _minimal_raw(format={"template": bad})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == [f"Invalid format template: {bad}"]


@pytest.mark.parametrize("template", sorted(VALID_FORMATS))
def test_all_valid_format_templates_accepted(tmp_path, template):
    raw = _minimal_raw(format={"template": template})
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


def test_missing_format_section_defaults_to_chatml(tmp_path):
    raw = _minimal_raw(format=None)
    assert _issues(_Cfg(raw, _abs_paths(tmp_path))) == []


# --- validate_and_report --------------------------------------------------

def test_validate_and_report_valid_prints_success(tmp_path, capsys):
    cfg = _Cfg(_minimal_raw(), _abs_paths(tmp_path))
    assert validate_and_report(cfg) is True
    assert capsys.readouterr().out == "[config] Configuration valid\n"


def test_validate_and_report_invalid_prints_each_issue(tmp_path, capsys):
    raw = _minimal_raw(
        train={"model_name": "", "max_seq_length": 0},
        sources={},
        clean={"max_chars_per_message": 0},
        format={"template": "bogus"},
    )
    cfg = _Cfg(raw, _abs_paths(tmp_path))
    assert validate_and_report(cfg) is False
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[config] Validation issues:"
    assert lines[1:] == [
        "  - Missing train.model_name",
        "  - Invalid max_seq_length: 0",
        "  - No sources configured",
        "  - Invalid max_chars_per_message: 0",
        "  - Invalid format template: bogus",
    ]


def test_validate_and_report_no_path_attribute_lists_all_paths(capsys):
    cfg = _DictCfg(_minimal_raw())
    assert validate_and_report(cfg) is False
    lines = capsys.readouterr().out.splitlines()
    assert lines[1:] == [f"  - Missing path: {p}" for p in REQUIRED_PATHS]
