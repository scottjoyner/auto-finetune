"""Real-time watcher for Neo4j trace data.

Polls the x1-370 Neo4j instance for new TraceGroups and ToolCalls,
and triggers the extraction pipeline when new data is detected.

Usage:
    python -m src.neo4j_watcher [--interval=60] [--min-new=10]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

from src.config import Config, load
from src.extract_neo4j_traces import _get_neo4j_config, _connect, _safe_neo4j_value


def _count_records(cfg: Config) -> tuple[int, int]:
    """Return (trace_group_count, tool_call_count) from Neo4j."""
    neo_cfg = _get_neo4j_config(cfg)
    driver, err = _connect(neo_cfg["uri"], neo_cfg["user"], neo_cfg["password"],
                           neo_cfg["database"], retries=2)
    if driver is None:
        return -1, -1
    try:
        with driver.session(database=neo_cfg["database"]) as s:
            tg = s.run("MATCH (g:TraceGroup) RETURN count(g) AS c").single()["c"]
            tc = s.run("MATCH (tc:ToolCall) RETURN count(tc) AS c").single()["c"]
            return tg, tc
    except Exception:
        return -1, -1
    finally:
        try:
            driver.close()
        except Exception:
            pass


def _read_state(state_file: str) -> dict:
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {"trace_groups": 0, "tool_calls": 0, "last_check": 0}


def _write_state(state_file: str, state: dict) -> None:
    try:
        with open(state_file, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def watch(cfg: Config, interval: int = 60, min_new: int = 10) -> None:
    """Poll Neo4j and trigger extraction when new data is detected."""
    state_file = os.path.join(cfg.path("raw_dir"), "neo4j-watcher-state.json")
    state = _read_state(state_file)

    print(f"[watcher] Starting Neo4j trace watcher (interval={interval}s, min_new={min_new})")
    print(f"[watcher] State: {state}")

    while True:
        try:
            tg, tc = _count_records(cfg)
            if tg < 0 or tc < 0:
                print(f"[watcher] Neo4j unreachable; retrying in {interval}s")
                time.sleep(interval)
                continue

            new_tg = tg - state.get("trace_groups", 0)
            new_tc = tc - state.get("tool_calls", 0)

            if new_tg >= min_new or new_tc >= min_new:
                print(f"[watcher] New data detected: +{new_tg} trace groups, +{new_tc} tool calls")
                print("[watcher] Triggering incremental extraction...")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "src.cli", "neo4j-traces", "--incremental"],
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        check=True, timeout=300,
                    )
                    print("[watcher] Extraction complete")
                except subprocess.TimeoutExpired:
                    print("[watcher] Extraction timed out")
                except subprocess.CalledProcessError as e:
                    print(f"[watcher] Extraction failed: {e}")

                # Update state
                state["trace_groups"] = tg
                state["tool_calls"] = tc
                state["last_check"] = time.time()
                _write_state(state_file, state)
            else:
                state["last_check"] = time.time()
                _write_state(state_file, state)

            time.sleep(interval)
        except KeyboardInterrupt:
            print("[watcher] Stopped by user")
            break
        except Exception as e:
            print(f"[watcher] Error: {e}; retrying in {interval}s")
            time.sleep(interval)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Neo4j trace watcher")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--min-new", type=int, default=10, help="Minimum new records to trigger extraction")
    args = parser.parse_args()

    cfg = load()
    watch(cfg, interval=args.interval, min_new=args.min_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
