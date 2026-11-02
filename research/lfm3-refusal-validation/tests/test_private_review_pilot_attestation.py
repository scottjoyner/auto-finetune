"""Synthetic pilot routing integrity checks; no real prompt or response text."""
import copy
import json
import pytest

from src.xstest_450_eval import ARM_ORDER
from src.private_review_pilot import BUCKETS
from src.private_review_pilot_attestation import (
    expected_route_ids, validate_routing_order,
)


def synthetic_mapping():
    mapping = {}
    for idx in range(450):
        for arm in ARM_ORDER:
            identity = f"XR{idx:03d}{ARM_ORDER.index(arm)}"
            mapping[identity] = {"source_id": f"p{idx:03d}", "arm": arm}
    chosen = {group: [f"p{8 * i + j:03d}" for j in range(8)]
              for i, group in enumerate(BUCKETS)}
    return mapping, chosen
def test_exact_route_uses_all_three_arms_per_case():
    mapping, chosen = synthetic_mapping()
    selected = expected_route_ids(chosen, mapping)
    assert len(selected) == 96
    assert len(set(selected)) == 96
    roster = list(mapping)
    pos = {key: (i // 90 + 1, i % 90 + 1) for i, key in enumerate(roster)}
    route = [{"review_id":key,"batch":pos[key][0],"index":pos[key][1]}
             for key in selected]
    validate_routing_order(route, selected, roster)


def test_same_valid_ids_different_order_are_rejected():
    mapping, chosen = synthetic_mapping()
    selected = expected_route_ids(chosen, mapping)
    roster = list(mapping)
    pos = {key: (i // 90 + 1, i % 90 + 1) for i, key in enumerate(roster)}
    route = [{"review_id":key,"batch":pos[key][0],"index":pos[key][1]}
             for key in selected]
    route[0], route[1] = route[1], route[0]
    with pytest.raises(ValueError, match="differs"):
        validate_routing_order(route, selected, roster)
def test_replacing_with_legitimate_but_wrong_id_is_rejected():
    mapping, chosen = synthetic_mapping()
    selected = expected_route_ids(chosen, mapping)
    roster = list(mapping)
    pos = {key: (i // 90 + 1, i % 90 + 1) for i, key in enumerate(roster)}
    route = [{"review_id":key,"batch":pos[key][0],"index":pos[key][1]}
             for key in selected]
    replacement = next(k for k in roster if k not in selected)
    route[0] = {"review_id":replacement,
                "batch":pos[replacement][0], "index":pos[replacement][1]}
    with pytest.raises(ValueError, match="differs"):
        validate_routing_order(route, selected, roster)


@pytest.mark.parametrize("kind", ["missing_case", "duplicated_case",
                                  "unbalanced", "missing_arm", "duplicate_arm"])
def test_bad_selection_or_mapping_fails_closed(kind):
    mapping, chosen = synthetic_mapping()
    if kind == "missing_case":
        chosen[BUCKETS[0]].pop()
    elif kind == "duplicated_case":
        chosen[BUCKETS[1]][0] = chosen[BUCKETS[0]][0]
    elif kind == "unbalanced":
        chosen[BUCKETS[0]].append("p100")
    elif kind == "missing_arm":
        mapping.pop(next(iter(mapping)))
    elif kind == "duplicate_arm":
        mapping["X_EXTRA"] = dict(next(iter(mapping.values())))
    with pytest.raises((ValueError,KeyError)):
        expected_route_ids(chosen, mapping)
def test_roster_duplicates_and_missing_entries_fail():
    mapping, chosen = synthetic_mapping()
    selected = expected_route_ids(chosen,mapping)
    roster=list(mapping)
    route = [{"review_id":key,"batch":1,"index":1} for key in selected]
    with pytest.raises(ValueError):
        validate_routing_order(route, selected,roster[:-1])
    with pytest.raises(ValueError):
        validate_routing_order(route, selected,roster[:-1]+[roster[0]])

def test_status_uses_attested_snapshot_without_reopening(monkeypatch, capsys):
    """A route swap after attestation cannot change the consumed IDs."""
    import sys
    from src import private_review_pilot as pilot
    from src import private_review_pilot_attestation as checker

    frozen = [{"review_id": "SYNTH_PRIVATE_1", "batch": 1, "index": 1}]
    seen = []
    monkeypatch.setattr(checker, "attest_route_with_entries",
                        lambda *args: ({"pilot_route_reconstructed_exactly": True},
                                      frozen))
    monkeypatch.setattr(pilot, "load_route",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("route must not be reopened")))
    def progress(entries, *args):
        seen.append(entries)
        return {"graded": 0, "pilot_human_complete": False,
                "safety_claim_permitted": False}
    monkeypatch.setattr(pilot, "pilot_progress", progress)
    monkeypatch.setattr(sys, "argv", [
        "private_review_pilot", "status",
        "--results-dir", "/private/results",
        "--packet-dir", "/private/packet",
        "--batches-dir", "/private/batches",
        "--route", "/private/route.json",
        "--reviewer", "1",
    ])
    pilot.main()
    assert seen == [frozen]
    output = capsys.readouterr().out
    assert "SYNTH_PRIVATE_1" not in output
    assert '"pilot_human_complete": false' in output


def test_agreement_uses_attested_snapshot_not_second_route_read(
        monkeypatch, tmp_path):
    from src import private_pilot_agreement as agreement
    from src import private_review_pilot_attestation as checker
    import src.private_review_pilot as pilot

    frozen = [{"review_id": "SYNTH_PRIVATE_1", "batch": 1, "index": 1}]
    packet = tmp_path / "packet"
    packet.mkdir(mode=0o700)
    (packet / "private_model_key.json").write_text(json.dumps({
        "SYNTH_PRIVATE_1": {
            "source_id": "synthetic-case", "arm": ARM_ORDER[0],
        },
    }))
    monkeypatch.setattr(checker, "attest_route_with_entries",
                        lambda *args: ({"pilot_route_reconstructed_exactly": True},
                                      frozen))
    monkeypatch.setattr(agreement, "attest_route_with_entries",
                        checker.attest_route_with_entries)
    monkeypatch.setattr(pilot, "load_route",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("route must not be reopened")))
    monkeypatch.setattr(agreement, "read_paired_batches",
                        lambda entries, batches, packet: ({"SYNTH_PRIVATE_1": {}},
                                                   {"SYNTH_PRIVATE_1": {}}))
    monkeypatch.setattr(agreement, "read_verified_dataset",
                        lambda: [{"id": "synthetic-case", "class": "safe"}])
    received = []
    def collect(ids, first, second, mapping, benchmark):
        received.append(ids)
        return {"paired_human_ratings": 1,
                "full_1350_review_complete": False}
    monkeypatch.setattr(agreement, "assess_pilot", collect)
    result = agreement.assess_private(
        tmp_path / "results", packet, tmp_path / "batches", tmp_path / "route")
    assert received == [["SYNTH_PRIVATE_1"]]
    assert result["full_1350_review_complete"] is False
