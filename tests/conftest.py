"""Shared pytest fixtures for the auto-finetune test suite."""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

# Make `src` importable as a package regardless of cwd.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import copy

from src.config import _DEFAULTS, Config  # noqa: E402


def make_cfg(**overrides):
    """Return a Config with a DEEP copy of defaults (safe to mutate)."""
    raw = copy.deepcopy(_DEFAULTS)
    for k, v in overrides.items():
        raw[k].update(v)
    return Config(raw=raw)


@pytest.fixture
def tmp_root(tmp_path):
    """A temp project root with raw/cleaned/datasets dirs + a config.yaml."""
    for d in ("raw", "cleaned", "datasets"):
        os.makedirs(tmp_path / "data" / d, exist_ok=True)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "paths:\n"
        "  raw_dir: data/raw\n"
        "  cleaned_dir: data/cleaned\n"
        "  dataset_dir: data/datasets\n"
        "sources:\n"
        "  opencode:\n"
        "    db_path: ''\n"
        "    extra_dbs: []\n"
        "  hermes:\n"
        "    dir: ''\n"
        "    enabled: false\n"
        "train:\n"
        "  model_name: Qwen/Qwen2.5-7B-Instruct\n"
    )
    return tmp_path


@pytest.fixture
def cfg(tmp_root):
    return Config(raw=dict(_DEFAULTS))


@pytest.fixture
def sample_session():
    """A normalized opencode-shaped session record (as written by extract)."""
    return {
        "source": "opencode",
        "session_id": "ses_demo",
        "title": "Demo session",
        "agent": "build",
        "model": "gpt-5.4-mini",
        "project_id": "proj1",
        "directory": "/home/x",
        "time_created": 1,
        "time_updated": 2,
        "messages": [
            {
                "id": "msg_u1", "role": "user", "agent": "build", "model": "gpt-5",
                "provider": "openai", "time": 10, "parent_id": None,
                "parts": [{"type": "text", "text": "Add a retry to fetch"}],
            },
            {
                "id": "msg_a1", "role": "assistant", "agent": "build", "model": "gpt-5",
                "provider": "openai", "time": 20, "parent_id": "msg_u1",
                "parts": [
                    {"type": "reasoning", "text": "I will edit the file."},
                    {"type": "tool", "tool": "edit", "call_id": "c1",
                     "status": "completed",
                     "input": {"file": "a.py", "old": "x", "new": "y"},
                     "output": "edited"},
                    {"type": "text", "text": "Done. Added retry logic."},
                    {"type": "patch", "hash": "abc", "files": ["a.py"]},
                ],
            },
        ],
    }


@pytest.fixture
def make_opencode_db(tmp_path):
    """Build a real SQLite DB shaped like opencode's, return its path."""
    def _make(corrupt_tail: bool = False, extra_sessions: int = 0) -> str:
        db_path = str(tmp_path / "opencode.db")
        con = sqlite3.connect(db_path)
        con.execute("""CREATE TABLE session(
            id text PRIMARY KEY, project_id text, workspace_id text, parent_id text,
            slug text, directory text, path text, title text, version text,
            share_url text, metadata text, cost real, tokens_input integer,
            tokens_output integer, tokens_reasoning integer, tokens_cache_read integer,
            tokens_cache_write integer, revert text, permission text, agent text,
            model text, time_created integer, time_updated integer,
            time_compacting integer, time_archived integer)""")
        con.execute("""CREATE TABLE message(
            id text PRIMARY KEY, session_id text, time_created integer,
            time_updated integer, data text)""")
        con.execute("""CREATE TABLE part(
            id text PRIMARY KEY, message_id text, session_id text,
            time_created integer, time_updated integer, data text)""")

        sid = "ses_demo"
        con.execute(
            "INSERT INTO session VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, "proj1", None, None, "slug", "/home/x", "p", "Demo", "v", "",
             "", 0, 0, 0, 0, 0, 0, None, None, "build", "gpt-5.4-mini", 1, 2, None, None),
        )
        for i in range(extra_sessions):
            con.execute(
                "INSERT INTO session VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"ses_{i}", "p", None, None, "s", "d", "", f"S{i}", "v", "", "",
                 0, 0, 0, 0, 0, 0, None, None, "build", "m", 10 + i, 20 + i, None, None),
            )

        con.execute("INSERT INTO message VALUES(?,?,?,?,?)",
                    ("msg_u1", sid, 10, 10,
                     json.dumps({"role": "user", "agent": "build",
                                 "model": {"providerID": "openai", "modelID": "gpt-5"},
                                 "time": {"created": 10}})))
        con.execute("INSERT INTO message VALUES(?,?,?,?,?)",
                    ("msg_a1", sid, 20, 20,
                     json.dumps({"role": "assistant", "agent": "build",
                                 "model": {"providerID": "openai", "modelID": "gpt-5"},
                                 "time": {"created": 20}, "parentID": "msg_u1"})))
        con.execute("INSERT INTO part VALUES(?,?,?,?,?,?)",
                    ("p0", "msg_u1", sid, 10, 10, json.dumps({"type": "text", "text": "Add retry"})))
        con.execute("INSERT INTO part VALUES(?,?,?,?,?,?)",
                    ("p1", "msg_a1", sid, 20, 20,
                     json.dumps({"type": "reasoning", "text": "thinking"})))
        con.execute("INSERT INTO part VALUES(?,?,?,?,?,?)",
                    ("p2", "msg_a1", sid, 20, 20,
                     json.dumps({"type": "tool", "tool": "edit", "callID": "c1",
                                 "state": {"status": "completed",
                                           "input": {"file": "a.py"}, "output": "ok"}})))
        con.execute("INSERT INTO part VALUES(?,?,?,?,?,?)",
                    ("p3", "msg_a1", sid, 20, 20,
                     json.dumps({"type": "text", "text": "Done."})))
        con.commit()
        con.close()

        if corrupt_tail:
            # Trailing junk page: forces MAX(rowid)/COUNT into the fallback path
            # of CorruptDB.iter_rows. The reader must still recover all rows.
            with open(db_path, "ab") as f:
                f.write(b"\x00" * 4096)
        return db_path
    return _make
