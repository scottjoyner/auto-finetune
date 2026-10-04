"""Typed, validating CLI flag parsing.

quantize.py, metrics.py and notify.py each hand-rolled ``int()``/``float()`` on
argv, and each raised a bare ValueError traceback on bad input -- so a typo like
``--bits=abc`` printed a stack trace instead of a message. This is the single
implementation they now share.

Two rules for callers:

* Print the returned error and return a non-zero exit code. Never let a parse
  failure become a silent default; ``_parse_int_flag`` in cli.py does exactly
  that (it returns None on a bad value, which reads as "flag not supplied") and
  should not be copied.
* Return ``(None, None)`` when the flag is absent -- absence is not an error,
  so callers can layer their own default on top.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

Coercer = Callable[[str], object]


def _type_name(cast: Coercer) -> str:
    return {int: "an integer", float: "a number"}.get(cast, cast.__name__)


def flag_value(argv: Iterable[str], name: str) -> str | None:
    """Raw text of --name=<value>, or None when absent."""
    prefix = name + "="
    for arg in argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


def parse_number(
    argv: Iterable[str],
    name: str,
    *,
    cast: Coercer = int,
    choices: tuple | set | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
):
    """Parse ``--name=<number>`` with validation.

    Returns ``(value, error)``. Exactly one is ever non-None:

    * ``(None, None)``  -- flag absent, caller applies its own default
    * ``(value, None)`` -- parsed and valid
    * ``(None, msg)``   -- present but invalid; ``msg`` is user-facing

    ``choices`` is enforced before the range checks, so an out-of-set value
    reports the set rather than a confusing bound.
    """
    raw = flag_value(argv, name)
    if raw is None:
        return None, None

    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return None, f"{name} must be {_type_name(cast)}, got {raw!r}"

    if choices is not None and value not in choices:
        return None, f"{name} must be one of {sorted(choices)}, got {value!r}"
    if minimum is not None and value < minimum:
        return None, f"{name} must be >= {minimum}, got {value!r}"
    if maximum is not None and value > maximum:
        return None, f"{name} must be <= {maximum}, got {value!r}"
    return value, None


def parse_text(
    argv: Iterable[str],
    name: str,
    *,
    choices: tuple | set | None = None,
    allow_empty: bool = True,
):
    """Parse ``--name=<text>`` with an optional allowed-value set.

    Returns ``(value, error)`` with the same contract as parse_number.
    """
    raw = flag_value(argv, name)
    if raw is None:
        return None, None
    if not allow_empty and not raw.strip():
        return None, f"{name} must not be empty"
    if choices is not None and raw not in choices:
        return None, f"{name} must be one of {sorted(choices)}, got {raw!r}"
    return raw, None
