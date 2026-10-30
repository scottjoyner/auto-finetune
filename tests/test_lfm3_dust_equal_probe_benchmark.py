"""One *actual* signed LFM3 model probe per forward-only comparator."""
import math
import pytest

from src.lfm3_dust_equal_probe_benchmark import (
    SEEDS, POLICIES, SIGMA, task_examples, candidate_set, choose, validate,
)


def test_registered_budget_and_new_task_partition():
    assert SEEDS==(25,26,27)
    assert POLICIES==("random_one_probe","neural_one_probe")
    for seed in SEEDS:
        validate(seed=seed)
        tasks=task_examples(seed)
        assert len(tasks["training"])==1
        assert len(tasks["diagnostic_same_family"])==4
        assert len(tasks["diagnostic_transfer"])==2
        assert len({e["input_key"] for values in tasks.values() for e in values})==7
        assert {e["family"] for e in tasks["training"]}=={"first_word"}
        assert {e["family"] for e in tasks["diagnostic_transfer"]}=={"uppercase"}
    for seed in (10,22,24,28):
        with pytest.raises(ValueError):
            validate(seed=seed)
    with pytest.raises(ValueError):
        validate(seed=25,probes_per_policy=8)
    with pytest.raises(ValueError):
        validate(seed=25,sigma=.1)


def test_signed_direction_set_contains_four_matched_antithetic_pairs():
    for seed in SEEDS:
        dirs=candidate_set(seed)
        assert len(dirs)==8
        for i in range(0,8,2):
            assert dirs[i]["pair"]==dirs[i+1]["pair"]==i//2
            assert dirs[i]["sign"]==1
            assert dirs[i+1]["sign"]==-1
            assert dirs[i]["direction"]==[-x for x in dirs[i+1]["direction"]]
            assert sum(x*x for x in dirs[i]["direction"])==pytest.approx(1,abs=1e-8)


def test_random_and_neural_rank_do_not_consume_any_new_gain_labels():
    scores=[.01,.06,.02,.03,-.01,.04,.02,.005]
    chosen=choose(25,scores)
    assert chosen["neural_one_probe"]==1
    assert chosen["random_one_probe"]==choose(25,list(scores))["random_one_probe"]
    assert len(chosen)==2
    with pytest.raises(ValueError):
        choose(25,[0.]*7)
    with pytest.raises(ValueError):
        choose(25,[float("nan")]+[0.]*7)
