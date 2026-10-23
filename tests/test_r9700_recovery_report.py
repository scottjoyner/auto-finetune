"""Offline security and fail-closed tests; do not contact Govee or x1."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from scripts import r9700_recovery_report as recovery
from tests.test_fleet_training_placement import healthy


def test_offline_no_remote_witness():
    report = recovery.assess({"state": "ON", "online": True}, None)
    assert report["status"] == "NO_GO"
    assert not report["may_launch_training"]
    assert "remote_evidence_unavailable" in report["reasons"]


def test_relay_on_is_not_proof_of_electrically_powered_gpu():
    data = healthy()
    data["oculink_identity"]["root_port_present"] = False
    report = recovery.assess({"state": "ON", "online": True}, data)
    assert "r9700_oculink_path_unverified" in report["reasons"]
    assert report["status"] == "NO_GO"


def test_offline_or_unknown_relay_denied():
    for relay in (
        {"state": "OFF", "online": True},
        {"state": "ON", "online": False},
        {"state": "UNKNOWN"},
        {},
    ):
        result = recovery.assess(relay, healthy())
        assert "plug_c_relay_not_confirmed_on" in result["reasons"]
        assert not result["may_launch_training"]


def test_all_green_is_operator_review_not_training_authority():
    result = recovery.assess({"state": "ON", "online": True}, healthy())
    assert result["status"] == "PHYSICAL_REVIEW_REQUIRED"
    assert result["operator_review_only"]
    assert result["may_launch_training"] is False


def test_no_mutating_url_allowed():
    for route in ("/devices/control", "/devices/state/turn?off=1", "/devices/state", "/wrong"):
        try:
            recovery._govee_get(route, "dummy-key")
        except ValueError as e:
            assert "read-only" in str(e)
        else:
            raise AssertionError("non GET-only route accepted")


def test_relay_evidence_is_minimal_and_secret_free(tmp_path, monkeypatch):
    key = tmp_path / "api-key"
    key.write_text("synthetic-secret-keep-private")
    key.chmod(0o600)
    monkeypatch.setattr(recovery, "KEY_PATH", key)
    calls = []

    def fake_request(route, secret):
        assert secret == "synthetic-secret-keep-private"
        calls.append(route)
        if route == "/devices":
            return {"data": {"devices": [
                {"device": recovery.PLUG_C, "model": "H5086", "deviceName": "towers"},
                {"device": "other", "model": "H5086", "deviceName": "private"}
            ]}}
        return {"data": {"properties": [{"powerState": "on"}, {"online": True}]}}

    monkeypatch.setattr(recovery, "_govee_get", fake_request)
    relay = recovery.read_relay()
    assert relay == {"state": "ON", "online": True, "reason": "provider_relay_state_only"}
    assert len(calls) == 2
    assert all("synthetic-secret-keep-private" not in c for c in calls)
    assert "synthetic-secret-keep-private" not in json.dumps(relay)
    assert "private" not in json.dumps(relay)


def test_bad_credential_permissions_fail_closed(tmp_path, monkeypatch):
    key = tmp_path / "api-key"
    key.write_text("never-leak")
    key.chmod(0o644)
    monkeypatch.setattr(recovery, "KEY_PATH", key)
    with patch.object(recovery, "_govee_get", side_effect=AssertionError("no API requests")):
        result = recovery.read_relay()
    assert result["state"] == "UNKNOWN"
    assert result["reason"] == "credential_permissions_unsafe"


def test_ambiguous_plug_identity_denied(tmp_path, monkeypatch):
    key = tmp_path / "api-key"
    key.write_text("synthetic-key")
    key.chmod(0o600)
    monkeypatch.setattr(recovery, "KEY_PATH", key)
    monkeypatch.setattr(recovery, "_govee_get",
                        lambda path, secret: {"data": {"devices": []}})
    assert recovery.read_relay()["reason"] == "plug_c_identity_not_unique"


def test_provider_redirect_never_forwards_credential():
    try:
        recovery._NoRedirect().redirect_request(
            None, None, 302, "Moved", {}, "https://untrusted.invalid/steal"
        )
    except ValueError as error:
        assert "refused" in str(error)
    else:
        raise AssertionError("Govee credential redirect not blocked")
