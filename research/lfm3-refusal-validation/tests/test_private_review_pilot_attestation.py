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
