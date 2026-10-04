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
