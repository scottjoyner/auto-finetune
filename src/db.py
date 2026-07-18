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
            # COUNT(*) aborts on a corrupt page; count via the resilient walk.
            n = 0
            for _ in self.iter_rows(table, ("rowid",), count_only=True):
                n += 1
            return n

    def _probe_max_rowid(self, table: str, hint: int = 0) -> int:
        """Find an upper bound on rowid by exponential probing.

        Used when both MAX(rowid) and COUNT(*) abort due to corruption. We grow
        a candidate ceiling exponentially until a large window contains no rows,
        so the linear walk in `iter_rows` covers every readable row.
        """
        def has_rows_between(lo: int, hi: int) -> bool:
            # single-rowid probes so one bad page can't abort the check
            step = max(1, (hi - lo) // 4096)
            r = lo
            while r <= hi:
                try:
                    if self._con.execute(
                        f"SELECT 1 FROM {table} WHERE rowid=?", (r,)
                    ).fetchone() is not None:
                        return True
                except Exception:
                    pass
                r += step
            return False

        ceiling = max(hint, 1024)
        # grow until a window above `ceiling` is empty
        while ceiling < (1 << 34):  # hard cap ~17e9 rowids
            hi = ceiling * 2
            if not has_rows_between(ceiling, hi):
                break
            ceiling = hi
        return ceiling

    def iter_rows(self, table: str, columns: tuple[str, ...], *,
                  where_rowid_in: bool = False, count_only: bool = False):
        """Yield rows one rowid at a time. On corruption, skip the bad row.

        Yields tuples in the order of `columns`. If `count_only`, yields ints
        (the rowid) so the caller can count readable rows.
        """
        col_sql = "rowid" if count_only else ", ".join(columns)
        maxid = self.max_rowid(table)
        if maxid == 0:
            # MAX(rowid) aborted on a corrupt page even though rows exist.
            # Probe exponentially for an upper bound instead of COUNT(*)
            # (which also aborts on the same corruption).
            maxid = self._probe_max_rowid(table)
            if maxid == 0:
                return
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
