"""CPU-free contracts for public, disjoint, no-promotion task probe."""
import pytest
from src.lfm3_lora_text_task_probe import _digest, _example, dataset, run, FAMILIES


def test_no_duplicate_examples_across_splits():
    for seed in (1, 2, 3):
        sections = dataset(seed)
        assert [len(sections[x]) for x in ("train", "selection", "untouched_test")] == [24, 9, 9]
        assert all(set(item) == {"family", "prompt", "answer", "foil", "input_key"}
                   for rows in sections.values() for item in rows)
        all_keys = [x["input_key"] for rows in sections.values() for x in rows]
        assert len(set(all_keys)) == len(all_keys)
        assert len({_digest(v) for v in sections.values()}) == 3
        assert all(x["answer"] != x["foil"] for rows in sections.values() for x in rows)
        assert all({x["family"] for x in rows} == set(FAMILIES) for rows in sections.values())


def test_seed_replay_same_hashes():
    for seed in (1, 2, 3):
        first, again = dataset(seed), dataset(seed)
        assert first == again
        assert {k: _digest(v) for k, v in first.items()} == {
            k: _digest(v) for k, v in again.items()}
    assert _digest(dataset(1)["train"]) != _digest(dataset(2)["train"])


def test_arithmetic_and_parity_answers_are_correct():
    assert _example("parity", 12, 8)["answer"] == "yes"
    assert _example("parity", 13, 8)["answer"] == "no"
    assert _example("greater", 19, 7)["answer"] == "yes"
    assert _example("greater", 3, 7)["answer"] == "no"
    assert _example("sum", 13, 7)["answer"] == "20"
    with pytest.raises(ValueError):
        _example("malformed", 3, 7)


@pytest.mark.parametrize("bad_seed", [-1, 2**32, True, "one"])
def test_invalid_seed_fails_closed(bad_seed):
    with pytest.raises(ValueError):
        dataset(bad_seed)


def test_bad_steps_and_optimizer_arms_block_before_imports():
    with pytest.raises(ValueError, match="bounded_steps"):
        run(seeds=(1,), steps=21, experimental_muon=True)
    with pytest.raises(ValueError, match="paired_text_probe"):
        run(seeds=(1,), steps=4, optimizers=("adamw",),
            experimental_muon=False)


def test_probe_requires_offline_cpu_and_lora_muon_optin(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="offline_cpu_isolation"):
        run(seeds=(1,), steps=1, experimental_muon=True)
    with pytest.raises(ValueError, match="experimental_muon"):
        run(seeds=(1,), steps=1, experimental_muon=False)


def test_short_curriculum_is_not_a_single_task_family():
    for seed in (1, 2, 3):
        early = [row["family"] for row in dataset(seed)["train"][:10]]
        assert set(early) == set(FAMILIES)
        assert early != sorted(early)
