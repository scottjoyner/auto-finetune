"""Hermetic guard: src.flags must stay stdlib-only.

src.flags is imported during early CLI startup (quantize/metrics/notify), so
it must never pull in a third-party dependency. This test parses the module
with the stdlib `ast` module only -- no pytest, no import of the module, no
network, no service, no DB -- so it runs anywhere Python 3.10+ runs.

This is additive: it introduces a new test file and touches no runtime,
security, compose, or gate files.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

SRC_FLAGS = Path(__file__).resolve().parents[1] / "src" / "flags.py"


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names imported anywhere in *path* (stdlib ast only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class FlagsStdlibOnly(unittest.TestCase):
    def test_flags_imports_are_stdlib_only(self):
        roots = _imported_roots(SRC_FLAGS)
        stdlib = set(sys.stdlib_module_names)
        third_party = sorted(r for r in roots if r not in stdlib)
        self.assertEqual(
            third_party, [],
            f"src.flags imports non-stdlib modules: {third_party}",
        )

    def test_flags_file_exists(self):
        self.assertTrue(SRC_FLAGS.is_file(), f"missing {SRC_FLAGS}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
