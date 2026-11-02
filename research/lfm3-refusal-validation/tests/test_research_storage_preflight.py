"""No filesystem writes, no cleanup, no model execution."""
import math
import pytest

from src.research_storage_preflight import (
    GIB, assess_bytes, preflight, stat_capacity,
)


def test_large_write_blocked_at_low_free_space():
    r=assess_bytes(9*GIB,5*GIB,8*GIB,20*GIB)
    assert not r["large_write_allowed"]
    assert r["required_available_gib"]==20
    assert r["available_gib"]==9


def test_floor_and_write_plus_reserve_both_enforced():
    assert assess_bytes(25*GIB,5*GIB,8*GIB,20*GIB)["large_write_allowed"]
    r=assess_bytes(25*GIB,19*GIB,8*GIB,20*GIB)
    assert not r["large_write_allowed"]
    assert r["required_available_gib"]==27
    assert assess_bytes(27*GIB,19*GIB,8*GIB,20*GIB)["large_write_allowed"]
@pytest.mark.parametrize("invalid", [-1,float("nan"),float("inf"),True, "1"])
def test_bad_capacity_inputs_fail_closed(invalid):
    with pytest.raises(ValueError):
        assess_bytes(invalid,5*GIB,8*GIB,20*GIB)


def test_preflight_read_only_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(
        "src.research_storage_preflight.stat_capacity",
        lambda _: 11*GIB)
    r=preflight(tmp_path,planned_write_gib=5)
    assert r["target"]==str(tmp_path)
    assert r["large_write_allowed"] is False
    assert r["writes_or_deletions_performed"] is False
    assert r["space_reserved"] is False
    with pytest.raises(ValueError):
        preflight(tmp_path,planned_write_gib=float("nan"))


def test_nonexistent_target_refused(tmp_path):
    with pytest.raises(ValueError):
        stat_capacity(tmp_path/"missing")


def test_plain_directory_is_not_claimed_as_nas(tmp_path):
    base=tmp_path/"root"
    nas=tmp_path/"pretend_nas"
    base.mkdir()
    nas.mkdir()
    with pytest.raises(ValueError,match="mounted"):
        preflight(base,nas_path=nas)
