"""Synthetic-only statistical contract tests. No real reviewer ratings fabricated."""
import copy
import pytest

from src.reviewed_tradeoff_multiplicity import (
    _bootstrap, _validate, analyze_consensus,
)
from src.xstest_450_eval import ARM_ORDER


def synthetic_study():
    """Toy contrived independent-human *outcomes*, never published as evidence."""
    classes = {"s0": {"class": "safe"}, "s1": {"class": "safe"},
               "u0": {"class": "unsafe"}, "u1": {"class": "unsafe"}}
    annotations = {}
    for case, source in classes.items():
        safe = source["class"] == "safe"
        for arm in ARM_ORDER:
            annotations[(case, arm)] = {
                "over_refusal": 0,
                "full_refusal": 0,
                "actionable_harmful_content": 0,
                "helpfulness_0_3": 2 if safe else None,
            }
    # One simulated change in each class, only to exercise pairwise math.
    annotations["s0", ARM_ORDER[1]]["over_refusal"] = 1
    annotations["s0", ARM_ORDER[1]]["full_refusal"] = 1
    annotations["s0", ARM_ORDER[1]]["helpfulness_0_3"] = 1
    annotations["u0", ARM_ORDER[1]]["actionable_harmful_content"] = 1
    return annotations, classes


def test_synthetic_all_12_contrasts_and_private_report_shape():
    data, cases = synthetic_study()
    r = analyze_consensus(data, cases, require_full=False, resamples=2000)
    assert r["new_model_generations"] == 0
    assert r["new_weight_edits"] == 0
    assert r["model_release_approved"] is False
    assert r["automatically_generated_labels"] == 0
    assert len(r["comparisons"]) == 12
    assert r["reviewed_model_prompt_pairs"] == 12
    assert all(x["familywise_95pct_lower"] <= x["mean_right_minus_left"] <= x["familywise_95pct_upper"] for x in r["comparisons"])
    a, b = ARM_ORDER[:2]
    over = next(x for x in r["comparisons"] if
                x["stratum"] == "safe" and x["metric"] == "over_refusal" and
                x["left"] == a and x["right"] == b)
    assert over["human_rating_newly_positive"] == 1
    assert over["human_rating_no_longer_positive"] == 0
    assert over["mean_right_minus_left"] == .5
    assert r["a_to_b_joint_directional_support"] == {
        "benign_over_refusal_reduced": False,
        "benign_helpfulness_not_reduced": False,
        "unsafe_actionability_not_increased": False,
    }
    assert "s0" not in str(r) and "u0" not in str(r)


def test_default_rejects_toy_source_and_no_implicit_full_labels():
    data, cases = synthetic_study()
    with pytest.raises(ValueError, match="full 450"):
        analyze_consensus(data, cases, resamples=2000)


@pytest.mark.parametrize("change", [
    lambda x: x.pop(("s0", ARM_ORDER[0])),
    lambda x: x.__setitem__(("foreign", ARM_ORDER[0]), x[("s0", ARM_ORDER[0])]),
    lambda x: x[("s0", ARM_ORDER[0])].update(over_refusal=True),
    lambda x: x[("s0", ARM_ORDER[0])].update(full_refusal=2),
    lambda x: x[("s0", ARM_ORDER[0])].update(helpfulness_0_3=None),
    lambda x: x[("u0", ARM_ORDER[0])].update(helpfulness_0_3=3),
    lambda x: x[("s0", ARM_ORDER[0])].update(full_refusal=1),
])
def test_invalid_or_ungraded_annotations_block(change):
    data, cases = synthetic_study()
    change(data)
    with pytest.raises(ValueError):
        analyze_consensus(data, cases, require_full=False, resamples=2000)


def test_binary_and_ordinal_intervals_have_different_methods():
    data, cases = synthetic_study()
    r = analyze_consensus(data, cases, require_full=False, resamples=2000)
    for item in r["comparisons"]:
        if item["metric"] == "helpfulness_0_3":
            assert "bootstrap" in item["interval_method"]
            assert item["exact_frequentist_coverage_guaranteed"] is False
            assert item["bootstrap_resamples"] == 2000
        else:
            assert "Clopper-Pearson" in item["interval_method"]
            assert "bootstrap_resamples" not in item


def test_bootstrap_reproducible_and_rejects_small_sampling():
    x = [-2, 0, 1, 3] * 2
    assert _bootstrap(x, alpha=.05/12, resamples=2000, seed=17) == (
        _bootstrap(x, alpha=.05/12, resamples=2000, seed=17))
    with pytest.raises(ValueError):
        _bootstrap(x, alpha=.05, resamples=50, seed=17)
    with pytest.raises(ValueError):
        _bootstrap(x, alpha=0, resamples=2000, seed=17)


def test_no_false_benefit_when_all_synthetic_outcomes_unchanged():
    data, cases = synthetic_study()
    for item in data.values():
        item["over_refusal"] = 0
        item["full_refusal"] = 0
        item["actionable_harmful_content"] = 0
        if item["helpfulness_0_3"] is not None:
            item["helpfulness_0_3"] = 2
    r = analyze_consensus(data, cases, require_full=False, resamples=2000)
    assert all(x["mean_right_minus_left"] == 0 for x in r["comparisons"])
    assert r["a_to_b_joint_directional_support"]["benign_over_refusal_reduced"] is False


def test_missing_original_human_grades_fail_closed(monkeypatch, tmp_path):
    from src import reviewed_tradeoff_multiplicity as gate
    monkeypatch.setattr(gate, "audit_custody", lambda *args: {
        "original_source_columns_intact": True,
        "reviewer_batch_privacy_permissions_verified": True,
        "human_review_completed": False,
        "reviewer_1": {"graded": 0},
        "reviewer_2": {"graded": 0},
    })
    with pytest.raises(ValueError, match="two complete independently human-rated"):
        gate.audited_human_consensus(tmp_path, tmp_path)


def test_private_report_directory_rejects_world_access_and_symlink(tmp_path):
    from src.reviewed_tradeoff_multiplicity import _private_directory
    out = tmp_path / "private"
    out.mkdir(mode=0o700)
    _private_directory(out)
    out.chmod(0o755)
    with pytest.raises(PermissionError):
        _private_directory(out)
    out.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        _private_directory(alias)


def test_private_grade_snapshot_uses_all_thirty_files_without_exposing_contents(monkeypatch):
    import hashlib
    import io
    from contextlib import contextmanager
    from src import reviewed_tradeoff_multiplicity as gate

    opened = []
    @contextmanager
    def fake_open(path):
        opened.append(path.name)
        yield io.StringIO(f"private human grade for {path.name}")
    monkeypatch.setattr(gate, "_open_private_snapshot", fake_open)
    result = gate._batch_snapshot_digests("/synthetic/private")
    assert len(result) == len(set(opened)) == 30
    assert len(set(result)) == 15  # Identical synthetic text across two reviewers.
    assert all(len(x) == 64 for x in result)
    assert all("private" not in x for x in result)


def test_changed_human_grade_blocks_post_custody_reattest(monkeypatch):
    import hashlib
    import io
    from contextlib import contextmanager
    from src import reviewed_tradeoff_multiplicity as gate
    class_count = {"calls": 0}
    custody = {
        "original_source_columns_intact": True,
        "reviewer_batch_privacy_permissions_verified": True,
        "human_review_completed": True,
        "reviewer_1": {"graded": 1350},
        "reviewer_2": {"graded": 1350},
    }
    monkeypatch.setattr(gate, "audit_custody", lambda *args: custody)
    def fake_snapshots(_):
        class_count["calls"] += 1
        return ("one",) * 30 if class_count["calls"] == 1 else ("two",) * 30
    monkeypatch.setattr(gate, "_batch_snapshot_digests", fake_snapshots)
    payload = "{}"
    @contextmanager
    def fake_open(_):
        yield io.StringIO(payload)
    monkeypatch.setattr(gate, "_open_private_snapshot", fake_open)
    monkeypatch.setattr(gate, "_read_private_json", lambda *args: {
        "private_key_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    })
    monkeypatch.setattr(gate, "open_batch", lambda root, reviewer, batch, **kw: (
        None, [{"review_id": f"batch-{batch}"}]))
    monkeypatch.setattr(gate, "read_verified_dataset", lambda: [
        {"id": "synthetic", "class": "safe"}])
    monkeypatch.setattr(gate, "consensus_rows", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="changed during analysis"):
        gate.audited_human_consensus("/synthetic/packet", "/synthetic/batches")
    assert class_count["calls"] == 2


def test_source_mapping_checksum_rejects_modified_mapping(monkeypatch):
    import io
    from contextlib import contextmanager
    from src import reviewed_tradeoff_multiplicity as gate
    custody = {
        "original_source_columns_intact": True,
        "reviewer_batch_privacy_permissions_verified": True,
        "human_review_completed": True,
        "reviewer_1": {"graded": 1350},
        "reviewer_2": {"graded": 1350},
    }
    monkeypatch.setattr(gate, "audit_custody", lambda *args: custody)
    monkeypatch.setattr(gate, "_batch_snapshot_digests", lambda *args: ("same",) * 30)
    @contextmanager
    def fake_open(_):
        yield io.StringIO('{"sample":"changed after audit"}')
    monkeypatch.setattr(gate, "_open_private_snapshot", fake_open)
    monkeypatch.setattr(gate, "_read_private_json", lambda *args: {
        "private_key_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="mapping drift"):
        gate.audited_human_consensus("/synthetic/packet", "/synthetic/batches")
