"""Coverage tests for extract_opencode.extract_db via a temp SQLite DB."""
from __future__ import annotations

import json
import sqlite3

from src import extract_opencode as E
from src.config import Config

SESSION_COLS = ("id", "project_id", "workspace_id", "parent_id", "slug", "directory",
                "path", "title", "version", "share_url", "metadata", "cost",
                "tokens_input", "tokens_output", "tokens_reasoning",
                "tokens_cache_read", "tokens_cache_write", "revert", "permission",
                "agent", "model", "time_created", "time_updated", "time_compacting",
                "time_archived")

CFG = Config({"extract": {"min_messages": 0, "exclude_agents": [], "include_agents": []}})


def _make_db(path: str, directory: str = "/proj/work") -> str:
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE session ({', '.join(SESSION_COLS)})")
    con.execute("CREATE TABLE message (id, session_id, time_created, time_updated, data)")
    con.execute("CREATE TABLE part (id, message_id, session_id, time_created, time_updated, data)")
    sess = ["sid"] + [""] * 24
    sess[1] = "p"          # project_id
    sess[5] = directory    # directory (index 5)
    sess[7] = "My Title"   # title (index 7)
    sess[19] = ""          # agent (index 19) - not excluded
    sess[20] = "m"         # model (index 20)
    sess[21] = 1           # time_created (index 21)
    sess[22] = 2           # time_updated (index 22)
    con.execute(f"INSERT INTO session VALUES ({','.join(['?']*25)})", tuple(sess))
    msg = {"role": "user", "agent": "a", "model": {"modelID": "m"},
           "time": {"created": 1}, "parentID": None}
    con.execute("INSERT INTO message VALUES (?,?,?,?,?)",
                (1, "sid", 1, 2, json.dumps(msg)))
    part = {"type": "text", "text": "hello"}
    con.execute("INSERT INTO part VALUES (?,?,?,?,?,?)",
                (1, 1, "sid", 1, 2, json.dumps(part)))
    con.commit()
    con.close()
    return path


def test_extract_db_writes_session(tmp_path):
    db = _make_db(str(tmp_path / "oc.db"))
    out = tmp_path / "out"
    out.mkdir()
    n = E.extract_db(CFG, db, str(out))
    assert n == 1
    rec = json.loads((out / "sid.json").read_text())
    assert rec["source"] == "opencode"
    assert rec["model"] == "m"
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][0]["parts"][0] == {"type": "text", "text": "hello"}


def test_extract_db_filtered_by_project(tmp_path):
    db = _make_db(str(tmp_path / "oc.db"), directory="/proj/work")
    out = tmp_path / "out"
    kept = E._extract_db_filtered(CFG, db, str(out), "work")
    assert kept == 1
    assert (out / "sid.json").exists()


def test_extract_db_filtered_excludes(tmp_path):
    db = _make_db(str(tmp_path / "oc.db"), directory="/other/place")
    out = tmp_path / "out"
    kept = E._extract_db_filtered(CFG, db, str(out), "work")
    assert kept == 0
