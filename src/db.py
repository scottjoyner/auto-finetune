"""Resilient reader for corrupt SQLite databases.

The migrated opencode DBs report `database disk image is malformed` on bulk
scans because a few pages are damaged. sqlite aborts an entire statement when it
hits a corrupt page, so we read rows **one rowid at a time** with a fresh
statement. A bad page then only costs the rows on that page instead of killing
the whole export.
"""
from __future__ import annotations

import apsw


class CorruptDB:
    """Minimal resilient wrapper around a (possibly corrupt) SQLite file."""

    def __init__(self, path: str):
        self.path = path
        # writable_schema lets us read past some structural inconsistencies.
        self._con = apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY)
        self._con.execute("PRAGMA writable_schema=ON")
        self._con.execute("PRAGMA busy_timeout=60000")

    def execute(self, sql: str, params: tuple = ()):
        return self._con.execute(sql, params)

    def table_exists(self, name: str) -> bool:
        row = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def max_rowid(self, table: str) -> int:
        try:
            return self._con.execute(f"SELECT MAX(rowid) FROM {table}").fetchone()[0] or 0
        except Exception:
            return 0

    def count(self, table: str) -> int:
        try:
            return self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            # fall back to rowid walk
            return self.iter_rows(table, ("rowid",), limit=None, count_only=True)

    def iter_rows(self, table: str, columns: tuple[str, ...], *,
                  where_rowid_in: bool = False, count_only: bool = False):
        """Yield rows one rowid at a time. On corruption, skip the bad row.

        Yields tuples in the order of `columns`. If `count_only`, yields ints
        (the rowid) so the caller can count readable rows.
        """
        col_sql = "rowid" if count_only else ", ".join(columns)
        maxid = self.max_rowid(table)
        if maxid == 0:
            # MAX(rowid) can abort on a corrupt tail page even when rows exist.
            # Fall back to the row count (which scans differently) as an upper
            # bound, then probe a margin past it to catch any high rowids.
            approx = self.count(table)
            if approx == 0:
                return
            # probe upward until we hit a long run of empty rowids
            maxid = approx
            probe = approx
            while probe < approx + 100000:
                probe += 1
                try:
                    hit = self._con.execute(
                        f"SELECT rowid FROM {table} WHERE rowid=?", (probe,)
                    ).fetchone()
                except Exception:
                    hit = None
                if hit is not None:
                    maxid = probe
        for r in range(1, maxid + 1):
            try:
                row = self._con.execute(
                    f"SELECT {col_sql} FROM {table} WHERE rowid=?", (r,)
                ).fetchone()
            except Exception:
                # corrupt page for this rowid; skip it.
                continue
            if row is None:
                continue
            if count_only:
                yield r
            else:
                yield row

    def close(self):
        try:
            self._con.close()
        except Exception:
            pass
