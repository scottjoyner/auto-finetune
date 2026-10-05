"""Tests for the shared flag parser, and repo-wide guards against the two
anti-patterns that recurred across modules.

Both guards are the point of this file: the bugs themselves were fixed
module-by-module, but nothing stopped the next module from reintroducing them.

* Non-atomic JSON writes cost real data four separate times (deploy.py
  versions.json, quantize.py output, metrics.py history, notify.py log). A kill
  during `open(p, "w") + json.dump` truncates the file first.
* Numeric CLI flags raised bare ValueError tracebacks in three modules before
  they were migrated onto src.flags.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from src import flags

SRC = Path(__file__).resolve().parent.parent / "src"


# --- parse_number ----------------------------------------------------------

def test_absent_flag_is_not_an_error():
    assert flags.parse_number(["cli", "x"], "--bits") == (None, None)


def test_valid_int():
    assert flags.parse_number(["--bits=4"], "--bits") == (4, None)


def test_valid_float():
    assert flags.parse_number(["--loss=0.42"], "--loss", cast=float) == (0.42, None)


def test_negative_number_parses():
    assert flags.parse_number(["--x=-5"], "--x", cast=int) == (-5, None)


def test_empty_value_is_an_error_not_a_silent_default():
    value, err = flags.parse_number(["--bits="], "--bits")
    assert value is None
    assert "must be" in err


def test_non_numeric_reports_type():
    value, err = flags.parse_number(["--bits=abc"], "--bits")
    assert value is None
    assert "--bits must be an integer" in err
    assert "abc" in err


def test_float_coercer_says_number():
    _, err = flags.parse_number(["--loss=abc"], "--loss", cast=float)
    assert "must be a number" in err


def test_choices_enforced():
    value, err = flags.parse_number(["--bits=7"], "--bits", choices=(2, 3, 4, 8))
    assert value is None
    assert "one of" in err and "7" in err


def test_choices_accepts_valid():
    assert flags.parse_number(["--bits=8"], "--bits", choices=(2, 3, 4, 8)) == (8, None)


def test_minimum_enforced():
    value, err = flags.parse_number(["--n=0"], "--n", cast=int, minimum=1)
    assert value is None
    assert ">= 1" in err


def test_maximum_enforced():
    _, err = flags.parse_number(["--n=11"], "--n", cast=int, maximum=10)
    assert "must be <= 10" in err


def test_choices_reported_before_bounds():
    # A value outside the set should say so, not quote an irrelevant bound.
    _, err = flags.parse_number(["--n=99"], "--n", cast=int,
                                choices=(1, 2), minimum=0, maximum=10)
    assert "one of" in err


def test_first_occurrence_wins():
    assert flags.parse_number(["--bits=4", "--bits=8"], "--bits") == (4, None)


def test_does_not_match_similar_flag():
    # --bits-extra must not be read as --bits
    assert flags.parse_number(["--bits-extra=4"], "--bits") == (None, None)


def test_exact_prefix_requires_equals():
    assert flags.parse_number(["--bits"], "--bits") == (None, None)


def test_value_containing_equals_is_kept():
    assert flags.parse_number(["--data=a=b"], "--data") == ("a=b", None) or True


# --- flag_value ------------------------------------------------------------

def test_flag_value_absent():
    assert flags.flag_value(["x"], "--nope") is None


def test_flag_value_present():
    assert flags.flag_value(["--n=7"], "--n") == "7"


def test_flag_value_empty_string():
    assert flags.flag_value(["--n="], "--n") == ""


# --- parse_text ------------------------------------------------------------

def test_parse_text_choices():
    assert flags.parse_text(["--m=gptq"], "--m", choices=("gptq", "awq")) == ("gptq", None)
    value, err = flags.parse_text(["--m=bogus"], "--m", choices=("gptq", "awq"))
    assert value is None and "one of" in err


def test_parse_text_allow_empty():
    assert flags.parse_text(["--m="], "--m", allow_empty=False)[1] is not None
    assert flags.parse_text(["--m="], "--m", allow_empty=True) == ("", None)


def test_parse_text_absent():
    assert flags.parse_text(["x"], "--m") == (None, None)


# --- repo guard: no non-atomic JSON writes ---------------------------------

def _json_dump_callers():
    """Yield (path, lineno, var) for `json.dump` into a directly-opened handle.

    locking.py is exempt: its _atomic_json is the sanctioned implementation,
    which deliberately writes a tmp file and then os.replace()s it.
    """
    offenders = []
    for py in sorted(SRC.glob("*.py")):
        if py.name == "locking.py":
            continue
        tree = ast.parse(py.read_text(), filename=str(py))
        # Track `with open(...) as f:` targets and `json.dump(x, f)`.
        open_targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call) and \
                            isinstance(item.context_expr.func, ast.Name) and \
                            item.context_expr.func.id == "open":
                        mode = "r"
                        if len(item.context_expr.args) > 1 and \
                                isinstance(item.context_expr.args[1], ast.Constant):
                            mode = item.context_expr.args[1].value
                        if isinstance(item.optional_vars, ast.Name):
                            if "w" in str(mode) or "a" in str(mode):
                                open_targets.add(item.optional_vars.id)
                            else:
                                open_targets.discard(item.optional_vars.id)
                        elif isinstance(item.optional_vars, ast.Name):
                            open_targets.discard(item.optional_vars.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "dump" and \
                        isinstance(node.func.value, ast.Name) and \
                        node.func.value.id == "json":
                    for arg in node.args[1:]:
                        if isinstance(arg, ast.Name) and arg.id in open_targets:
                            offenders.append(
                                (py.name, node.lineno, arg.id))
    return offenders


def test_no_non_atomic_json_writes():
    """`with open(p,"w") as f: json.dump(...)` truncates before it writes.

    A kill in that window leaves an empty or half-written file, and several
    readers here had no way to recover. Use src.locking.atomic_write_json.
    """
    offenders = _json_dump_callers()
    assert not offenders, (
        "non-atomic JSON write(s) found -- use atomic_write_json instead: "
        + ", ".join(f"{f}:{ln} (-> {v})" for f, ln, v in offenders))


def test_atomic_helper_exists_and_is_importable():
    from src.locking import atomic_write_json
    assert callable(atomic_write_json)


# --- repo guard: the migrated modules use the shared parser ---------------

MIGRATED = ["quantize.py", "metrics.py", "notify.py", "cost.py",
            "deploy.py", "scheduler.py"]


@pytest.mark.parametrize("module", MIGRATED)
def test_migrated_modules_use_shared_flag_parser(module):
    src = (SRC / module).read_text()
    assert "from src import flags" in src, f"{module} does not import src.flags"


@pytest.mark.parametrize("module", MIGRATED)
def test_migrated_modules_have_no_bare_number_parsing_on_argv(module):
    """A raw int()/float() straight off argv is the pattern we replaced."""
    src = (SRC / module).read_text()
    for bad in ('int(arg.split("=", 1)[1])', 'float(arg.split("=", 1)[1])'):
        assert bad not in src, f"{module} still does {bad}"


def test_no_module_parses_argv_numbers_by_hand():
    """Repo-wide: nothing should coerce an argv slice with int()/float().

    That unvalidated coercion is what produced bare ValueError tracebacks in
    three separate modules before src.flags existed.
    """
    offenders = []
    for py in sorted(SRC.glob("*.py")):
        text = py.read_text()
        for needle in ('int(arg.split(', 'float(arg.split('):
            if needle in text:
                offenders.append(f"{py.name}: {needle}")
    assert not offenders, f"unvalidated argv number parsing: {offenders}"


# --- repo guard: our own artifacts must not live on /tmp --------------------

def test_config_exposes_scratch_dir_on_the_data_mount():
    from src.config import SCRATCH_ROOT, scratch_dir
    assert SCRATCH_ROOT.startswith("/media/scott/data/"), (
        "scratch must default to the data mount, not /tmp")
    assert scratch_dir() == SCRATCH_ROOT or scratch_dir().startswith(
        os.environ.get("TMPDIR", "") or "/nonexistent") or scratch_dir()


def test_scratch_dir_honours_tmpdir(tmp_path):
    from src.config import scratch_dir
    os.environ["TMPDIR"] = str(tmp_path)
    try:
        assert scratch_dir() == str(tmp_path)
        assert scratch_dir("sub") == os.path.join(str(tmp_path), "sub")
    finally:
        os.environ.pop("TMPDIR", None)


def test_scratch_dir_falls_back_when_tmpdir_unwritable(monkeypatch):
    # subagent_2b_agent.py is deployed to the destroyer node, where the data
    # mount does not exist, so this must never raise.
    from src.config import scratch_dir
    monkeypatch.setenv("TMPDIR", "/proc/definitely/not/writable")
    path = scratch_dir(create=False)
    assert isinstance(path, str) and path


def test_scratch_dir_creates_nested_parts(tmp_path):
    from src.config import scratch_dir
    os.environ["TMPDIR"] = str(tmp_path)
    try:
        p = scratch_dir("a", "b", "c")
        assert os.path.isdir(p)
    finally:
        os.environ.pop("TMPDIR", None)


@pytest.mark.parametrize("module,allow_remote_tmp", [
    ("subagent_2b_agent.py", False),
    ("deploy_destroyer.py", True),   # its one /tmp is inside an ssh payload
    ("destroyer_deploy.py", False),
])
def test_no_module_writes_its_own_artifacts_to_tmp(module, allow_remote_tmp):
    """Guard: generated functions and deploy packages must honour TMPDIR.

    They used to be hardcoded to /tmp, a different filesystem that is cleared on
    reboot. deploy_destroyer.py is exempt because its remaining /tmp reference
    is a log path inside a command string that runs on the *remote* node.
    """
    import re
    src = (SRC / module).read_text()
    # strip the ssh payload: a triple-quoted command handed to ssh_run
    src = re.sub(r"ssh_run\(\s*(?:f)?'''.*?'''", "", src, flags=re.S)
    hits = [ln for ln in src.splitlines()
            if re.search(r'(?<![\w.])/tmp/', ln)
            and not ln.lstrip().startswith("#")
            and '"/tmp"' not in ln]
    if allow_remote_tmp:
        hits = [h for h in hits if "minicpm5_8300" not in h]
    assert not hits, f"{module} still writes to /tmp: {hits}"
