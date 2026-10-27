"""Change Data Capture (CDC) for Neo4j.

Instead of polling for changes, uses Neo4j's built-in CDC capability
to stream changes in real-time. Falls back to polling if CDC is unavailable.

Neo4j CDC requires:
  - Neo4j 5.x with CDC enabled
  - dbms.cdc.enabled=true in neo4j.conf
  - A CDC consumer (this module)

Usage:
    python -m src.neo4j_cdc [--mode=stream|poll] [--interval=10]
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Generator

from src.config import Config, load
from src.extract_neo4j_traces import _get_neo4j_config, _connect, _safe_neo4j_value


class Neo4jCDC:
    """Change Data Capture consumer for Neo4j."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.neo_cfg = _get_neo4j_config(cfg)
        self._driver = None
        self._last_change_id = None

    def connect(self) -> bool:
        """Connect to Neo4j."""
        driver, err = _connect(
            self.neo_cfg["uri"], self.neo_cfg["user"],
            self.neo_cfg["password"], self.neo_cfg["database"], retries=3,
        )
        if driver is None:
            return False
        self._driver = driver
        return True

    def is_cdc_available(self) -> bool:
        """Check if CDC is enabled on the Neo4j instance."""
        if self._driver is None:
            return False
        try:
            with self._driver.session(database=self.neo_cfg["database"]) as s:
                # Check if CDC procedures are available
                result = s.run("CALL dbms.cdc.current() YIELD id RETURN id LIMIT 1")
                result.single()
                return True
        except Exception:
            return False

    def stream_changes(self, callback: Callable[[dict], None],
                       poll_interval: float = 5.0) -> None:
        """Stream changes from Neo4j.

        If CDC is available, uses real-time streaming.
        Otherwise, falls back to polling-based change detection.
        """
        if self.is_cdc_available():
            print("[cdc] CDC available; using real-time streaming")
            self._stream_cdc(callback)
        else:
            print("[cdc] CDC not available; falling back to polling")
            self._poll_changes(callback, poll_interval)

    def _stream_cdc(self, callback: Callable[[dict], None]) -> None:
        """Stream changes using Neo4j CDC."""
        print("[cdc] Starting CDC stream...")
        try:
            with self._driver.session(database=self.neo_cfg["database"]) as s:
                # Query for changes since last known ID
                query = """
                    CALL dbms.cdc.query($fromId) YIELD id, event, metadata
                    RETURN id, event, metadata
                    ORDER BY id
                """
                while True:
                    try:
                        result = s.run(query, fromId=self._last_change_id)
                        changes = list(result)
                        if changes:
                            for change in changes:
                                change_data = {
                                    "id": _safe_neo4j_value(change["id"]),
                                    "event": _safe_neo4j_value(change["event"]),
                                    "metadata": _safe_neo4j_value(change["metadata"]),
                                }
                                callback(change_data)
                                self._last_change_id = change["id"]
                        else:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("[cdc] Stopped by user")
                        break
                    except Exception as e:
                        print(f"[cdc] Error: {e}; retrying in 5s")
                        time.sleep(5)
        finally:
            if self._driver:
                self._driver.close()

    def _poll_changes(self, callback: Callable[[dict], None],
                      poll_interval: float = 5.0) -> None:
        """Fallback: poll for new records based on count changes."""
        print(f"[cdc] Polling every {poll_interval}s...")
        last_tg_count = 0
        last_tc_count = 0

        try:
            while True:
                try:
                    with self._driver.session(database=self.neo_cfg["database"]) as s:
                        tg = s.run("MATCH (g:TraceGroup) RETURN count(g) AS c").single()["c"]
                        tc = s.run("MATCH (tc:ToolCall) RETURN count(tc) AS c").single()["c"]

                        if tg > last_tg_count or tc > last_tc_count:
                            callback({
                                "type": "new_records",
                                "new_trace_groups": tg - last_tg_count,
                                "new_tool_calls": tc - last_tc_count,
                                "total_trace_groups": tg,
                                "total_tool_calls": tc,
                            })
                            last_tg_count = tg
                            last_tc_count = tc
                except Exception as e:
                    print(f"[cdc] Poll error: {e}")

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("[cdc] Stopped by user")
        finally:
            if self._driver:
                self._driver.close()

    def close(self) -> None:
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Neo4j Change Data Capture")
    parser.add_argument("--mode", choices=["stream", "poll"], default="poll")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval")
    args = parser.parse_args(argv)

    cfg = load()
    cdc = Neo4jCDC(cfg)
    if not cdc.connect():
        print("[cdc] Failed to connect to Neo4j")
        return 1

    print(f"[cdc] CDC available: {cdc.is_cdc_available()}")

    def on_change(change):
        print(f"[cdc] Change detected: {change}")

    cdc.stream_changes(on_change, poll_interval=args.interval)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
