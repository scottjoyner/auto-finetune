"""Tests for src.db (resilient corrupt-SQLite reader)."""
from __future__ import annotations

import apsw
import pytest

from src.db import CorruptDB


def test_table_exists(make_opencode_db):
    db = CorruptDB(make_opencode_db())
    assert db.table_exists("session") is True
    assert db.table_exists("nope") is False
    db.close()


def test_max_rowid_and_count(make_opencode_db):
    db = CorruptDB(make_opencode_db())
    assert db.max_rowid("session") == 1
    assert db.count("session") == 1
    assert db.max_rowid("part") == 4
    assert db.count("part") == 4
    db.close()


def test_iter_rows_returns_all(make_opencode_db):
    db = CorruptDB(make_opencode_db())
    rows = list(db.iter_rows("message", ("id", "session_id")))
    assert len(rows) == 2
    ids = {r[0] for r in rows}
    assert ids == {"msg_u1", "msg_a1"}
    db.close()


def test_iter_rows_count_only(make_opencode_db):
    db = CorruptDB(make_opencode_db())
    rowids = list(db.iter_rows("part", ("id",), count_only=True))
    assert rowids == [1, 2, 3, 4]
    db.close()


def test_iter_rows_skips_corrupt_tail(make_opencode_db):
    # trailing junk page must not drop valid rows
    db = CorruptDB(make_opencode_db(corrupt_tail=True))
    rows = list(db.iter_rows("part", ("id",)))
    assert len(rows) == 4
    db.close()


def test_iter_rows_empty_table(tmp_path):
    p = str(tmp_path / "empty.db")
    con = apsw.Connection(p)
    con.execute("CREATE TABLE t(id text, v text)")
    con.close()
    db = CorruptDB(p)
    assert list(db.iter_rows("t", ("id",))) == []
    db.close()


def test_iter_rows_injects_errors(make_opencode_db, monkeypatch):
    """Simulate a corrupt page mid-scan; reader must skip and continue."""
    real_db = CorruptDB(make_opencode_db())
    real_execute = real_db._con.execute
    calls = {"n": 0}

    class FlakyCon:
        def execute(self, sql, params=()):
            calls["n"] += 1
            if calls["n"] == 3:
                raise apsw.CorruptError("simulated corruption")
            return real_execute(sql, params)

    db = CorruptDB.__new__(CorruptDB)
    db._con = FlakyCon()
    db.path = real_db.path
    rows = list(db.iter_rows("part", ("id",)))
    assert len(rows) == 3  # one skipped
    real_db.close()


def test_execute_passthrough(make_opencode_db):
    db = CorruptDB(make_opencode_db())
    row = db.execute("SELECT COUNT(*) FROM session").fetchone()
    assert row[0] == 1
    db.close()


def test_close_is_safe_on_bad_conn():
    db = CorruptDB.__new__(CorruptDB)
    db._con = None
    db.close()  # must not raise


def test_max_rowid_returns_zero_on_error(make_opencode_db, monkeypatch):
    db = CorruptDB(make_opencode_db())
    class C:
        def execute(self, sql, params=()):
            raise apsw.CorruptError("boom")
    db._con = C()
    assert db.max_rowid("part") == 0


def test_iter_rows_recovers_with_maxrowid_zero(make_opencode_db, monkeypatch):
    """When MAX(rowid) errors (returns 0) but rows exist, fall back to count."""
    db = CorruptDB(make_opencode_db())
    monkeypatch.setattr(db, "max_rowid", lambda table: 0)
    rows = list(db.iter_rows("part", ("id",)))
    assert len(rows) == 4
