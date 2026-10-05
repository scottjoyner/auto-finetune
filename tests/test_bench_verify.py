"""Tests for benchmark verification and scoring.

verify_check decides whether a model passed a task, and those results feed the
candidate promotion manifest, so a scoring defect here promotes the wrong model.
The bug this file was written for: the command_exit/command_output checks ran
with no timeout, so one slow check wedged the entire benchmark -- and
bench_suite is what the scheduler's candidate pipeline calls to build the
manifest, so nothing above it had a timeout either.
"""
from __future__ import annotations

import time

import pytest

from src.bench import (
    CHECK_COMMAND_TIMEOUT,
    Task,
    _check_timeout,
    format_tool_result,
    parse_tool_results,
    verify_check,
    verify_task,
)


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sb"
    d.mkdir()
    return d


# --- timeouts --------------------------------------------------------------

def test_command_exit_has_a_default_timeout():
    assert CHECK_COMMAND_TIMEOUT > 0


def test_command_exit_times_out(sandbox):
    t0 = time.time()
    r = verify_check(sandbox, {"kind": "command_exit", "cmd": "sleep 30",
                               "expect_code": 0, "timeout": 2})
    elapsed = time.time() - t0
    assert r.passed is False
    assert elapsed < 15, f"no timeout applied (took {elapsed:.1f}s)"
    assert "timed out" in r.detail


def test_command_output_times_out(sandbox):
    t0 = time.time()
    r = verify_check(sandbox, {"kind": "command_output", "cmd": "sleep 30",
                               "expect": "x", "timeout": 2})
    assert r.passed is False
    assert time.time() - t0 < 15
    assert "timed out" in r.detail


def test_timeout_is_fail_closed(sandbox):
    # A check that cannot finish must not be reported as a pass.
    r = verify_check(sandbox, {"kind": "command_exit", "cmd": "sleep 30",
                               "expect_code": 0, "timeout": 2})
    assert r.passed is False


def test_per_check_timeout_overrides(sandbox):
    assert _check_timeout({"timeout": 5}) == 5.0


def test_garbage_timeout_falls_back_to_default(sandbox):
    assert _check_timeout({"timeout": "abc"}) == float(CHECK_COMMAND_TIMEOUT)
    assert _check_timeout({"timeout": None}) == float(CHECK_COMMAND_TIMEOUT)
    assert _check_timeout({"timeout": -1}) == float(CHECK_COMMAND_TIMEOUT)
    assert _check_timeout({}) == float(CHECK_COMMAND_TIMEOUT)


def test_slow_command_does_not_block_verify_task(sandbox):
    # The whole task must complete, not just one check.
    t0 = time.time()
    passed, total, detail = verify_task(sandbox, [
        {"kind": "command_exit", "cmd": "sleep 30", "timeout": 2},
        {"kind": "file_exists", "path": "nope.txt"},
    ])
    assert (passed, total) == (0, 2)
    assert time.time() - t0 < 20


# --- file checks -----------------------------------------------------------

def test_file_exists(sandbox):
    (sandbox / "a.txt").write_text("hi")
    assert verify_check(sandbox, {"kind": "file_exists", "path": "a.txt"}).passed
    assert not verify_check(sandbox, {"kind": "file_exists", "path": "b.txt"}).passed


def test_dir_exists(sandbox):
    (sandbox / "d").mkdir()
    assert verify_check(sandbox, {"kind": "dir_exists", "path": "d"}).passed
    assert not verify_check(sandbox, {"kind": "dir_exists", "path": "a.txt"}).passed


def test_file_contains(sandbox):
    (sandbox / "a.txt").write_text("hello world")
    assert verify_check(sandbox, {"kind": "file_contains", "path": "a.txt",
                                  "expect": "world"}).passed
    assert not verify_check(sandbox, {"kind": "file_contains", "path": "a.txt",
                                      "expect": "absent"}).passed


def test_file_contains_on_missing_file_is_false(sandbox):
    r = verify_check(sandbox, {"kind": "file_contains", "path": "gone.txt",
                               "expect": "x"})
    assert r.passed is False


def test_file_regex(sandbox):
    (sandbox / "a.txt").write_text("value = 42")
    assert verify_check(sandbox, {"kind": "file_regex", "path": "a.txt",
                                  "pattern": r"=\s*\d+"}).passed
    assert not verify_check(sandbox, {"kind": "file_regex", "path": "a.txt",
                                      "pattern": r"=\s*abc"}).passed


def test_invalid_regex_fails_closed(sandbox):
    (sandbox / "a.txt").write_text("x")
    r = verify_check(sandbox, {"kind": "file_regex", "path": "a.txt",
                               "pattern": "([unclosed"})
    assert r.passed is False
    assert "error" in r.detail


# --- command checks --------------------------------------------------------

def test_command_exit_match(sandbox):
    assert verify_check(sandbox, {"kind": "command_exit", "cmd": "true"}).passed
    assert not verify_check(sandbox, {"kind": "command_exit",
                                      "cmd": "false"}).passed


def test_command_exit_custom_code(sandbox):
    assert verify_check(sandbox, {"kind": "command_exit", "cmd": "exit 7",
                                  "expect_code": 7}).passed
    assert not verify_check(sandbox, {"kind": "command_exit", "cmd": "exit 0",
                                      "expect_code": 7}).passed


def test_command_runs_in_sandbox(sandbox):
    (sandbox / "marker.txt").write_text("x")
    r = verify_check(sandbox, {"kind": "command_output", "cmd": "ls",
                               "expect": "marker.txt"})
    assert r.passed is True


def test_command_output_match(sandbox):
    assert verify_check(sandbox, {"kind": "command_output", "cmd": "echo hello",
                                  "expect": "hello"}).passed
    assert not verify_check(sandbox, {"kind": "command_output", "cmd": "echo hello",
                                      "expect": "goodbye"}).passed


def test_bad_command_is_false_not_a_crash(sandbox):
    r = verify_check(sandbox, {"kind": "command_exit",
                               "cmd": "definitely_not_a_real_binary_xyz"})
    assert r.passed is False


# --- unknown / malformed checks --------------------------------------------

def test_unknown_kind_is_false(sandbox):
    r = verify_check(sandbox, {"kind": "no_such_kind"})
    assert r.passed is False
    assert "unknown check kind" in r.detail


def test_missing_kind_is_false(sandbox):
    assert verify_check(sandbox, {}).passed is False


def test_missing_path_key_is_false(sandbox):
    # KeyError is caught and reported as a failed check, not raised.
    r = verify_check(sandbox, {"kind": "file_exists"})
    assert r.passed is False


# --- verify_task -----------------------------------------------------------

def test_verify_task_counts_all_checks(sandbox):
    (sandbox / "a.txt").write_text("hello")
    passed, total, detail = verify_task(sandbox, [
        {"kind": "file_exists", "path": "a.txt"},
        {"kind": "file_exists", "path": "missing.txt"},
    ])
    assert (passed, total) == (1, 2)
    assert len(detail) == 2
    assert detail[0]["passed"] is True and detail[1]["passed"] is False


def test_verify_task_zero_checks(sandbox):
    assert verify_task(sandbox, []) == (0, 0, [])


# --- tool result formatting ------------------------------------------------

def test_format_tool_result_finetune_variant():
    assert format_tool_result("out", "finetune") == "<tool_result>out</tool_result>"


def test_format_tool_result_base_variant():
    assert format_tool_result("out", "base") == "<tool_response>\nout\n</tool_response>"


def test_parse_tool_results_extracts_and_strips():
    text = "before <tool_result>  a  </tool_result> mid <tool_result>b</tool_result>"
    assert parse_tool_results(text) == ["a", "b"]


def test_parse_tool_results_none_present():
    assert parse_tool_results("no results here") == []


def test_parse_tool_results_ignores_unclosed():
    assert parse_tool_results("<tool_result>dangling") == []


# --- Task ------------------------------------------------------------------

def test_result_with_no_checks_is_not_a_pass():
    # Fail closed: a task with zero checks must never score as success, or a
    # malformed task definition would silently count as a win.
    from src.bench import TaskResult
    r = TaskResult(task_id="t", kind="exec", model="m", runner="self")
    assert r.success is False
    r.checks_total = 2
    r.checks_passed = 2
    assert r.success is True
    r.checks_passed = 1
    assert r.success is False


def test_partial_checks_are_not_a_pass():
    from src.bench import TaskResult
    r = TaskResult(task_id="t", kind="exec", model="m", runner="self",
                   checks_passed=2, checks_total=3)
    assert r.success is False


def test_task_from_dict_defaults():
    t = Task.from_dict({"id": "t1", "prompt": "do it"})
    assert t.id == "t1"
    assert t.kind == "exec"
    assert t.checks == []
    assert t.max_turns == 8


def test_task_from_dict_reads_checks():
    t = Task.from_dict({"id": "t", "prompt": "p",
                        "checks": [{"kind": "file_exists", "path": "a"}]})
    assert len(t.checks) == 1


def test_check_timeout_is_module_constant():
    assert isinstance(CHECK_COMMAND_TIMEOUT, (int, float))
