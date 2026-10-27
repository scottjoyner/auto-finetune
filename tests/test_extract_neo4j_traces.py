"""Tests for src.extract_neo4j_traces pure-logic helpers."""
from __future__ import annotations

from src import extract_neo4j_traces as T


def test_safe_json_dict():
    assert T._safe_json('{"a": 1}') == {"a": 1}
    assert T._safe_json(None) is None
    assert T._safe_json("not json") == "not json"
    assert T._safe_json({"already": "dict"}) == {"already": "dict"}


def test_safe_str():
    assert T._safe_str(None) == ""
    assert T._safe_str("hello") == "hello"
    assert T._safe_str(123) == "123"


def test_safe_neo4j_value_datetime():
    # Simulate a Neo4j DateTime-like object
    class FakeDateTime:
        def __str__(self):
            return "2026-08-24T03:29:03.811Z"
    assert T._safe_neo4j_value(FakeDateTime()) == "2026-08-24T03:29:03.811Z"


def test_safe_neo4j_value_primitives():
    assert T._safe_neo4j_value(None) is None
    assert T._safe_neo4j_value("hello") == "hello"
    assert T._safe_neo4j_value(42) == 42
    assert T._safe_neo4j_value(True) is True
    assert T._safe_neo4j_value([1, 2, 3]) == [1, 2, 3]
    assert T._safe_neo4j_value({"a": 1}) == {"a": 1}


def test_trace_group_to_session_basic():
    group = {"correlation_id": "abc-123", "current_state": "completed"}
    events = [
        {"event_id": "e1", "event_type": "router.route_decision", "source": "auto-router",
         "payload_json": '{"candidates": [{"model": "qwen3.5-2b"}]}', "ts": "2026-08-24T00:00:00Z"},
        {"event_id": "e2", "event_type": "router.execution_stage.completed", "source": "auto-router",
         "payload_json": '{"status": "ok"}', "ts": "2026-08-24T00:00:01Z"},
    ]
    rec = T._trace_group_to_session(group, events)
    assert rec is not None
    assert rec["source"] == "neo4j-traces"
    assert rec["session_id"] == "trace_abc-123"
    assert len(rec["messages"]) == 2


def test_trace_group_to_session_too_few_events():
    group = {"correlation_id": "abc-123"}
    events = [{"event_id": "e1", "event_type": "test", "source": "s",
               "payload_json": "{}", "ts": "t1"}]
    assert T._trace_group_to_session(group, events) is None


def test_trace_group_to_session_no_corr_id():
    group = {"correlation_id": ""}
    events = [{"event_id": "e1", "event_type": "test", "source": "s",
               "payload_json": "{}", "ts": "t1"},
              {"event_id": "e2", "event_type": "test2", "source": "s",
               "payload_json": "{}", "ts": "t2"}]
    assert T._trace_group_to_session(group, events) is None


def test_get_neo4j_config_defaults():
    from src.config import Config
    cfg = Config(raw={})
    neo = T._get_neo4j_config(cfg)
    assert "uri" in neo
    assert "database" in neo
    assert neo["database"] == "assistx"
    assert "password" in neo


def test_get_neo4j_config_from_env(monkeypatch):
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", "test-pw")
    from src.config import Config
    cfg = Config(raw={})
    neo = T._get_neo4j_config(cfg)
    assert neo["password"] == "test-pw"
