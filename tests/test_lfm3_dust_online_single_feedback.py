"""One-label forward-only learned Q/K head may use past training probes ONLY."""
import json
from pathlib import Path
import pytest

from src.lfm3_dust_task_shift_probe import rebuild_training_head
from src.lfm3_dust_online_single_feedback import OnlineSingleFeedbackHead
from src.lfm3_dust_online_one_feedback_trial import SEEDS, tasks, candidates, policy_choice

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"research/evidence/lfm3-dust-text-transfer-20261029"


def new_controller(seed=28):
    old=(10,11,12)[SEEDS.index(seed)]
    source=json.loads((SOURCE/f"seed{old}-extraction-to-uppercase-18episodes.json").read_text())
    return OnlineSingleFeedbackHead(rebuild_training_head(source),seed=seed),source


def test_sequential_single_feedback_changes_only_head_and_not_training_authority():
    ctrl,source=new_controller()
    direction=[1.]+[0.]*7
    context=source["events"][0]["context"]
    original=ctrl.head.parameter_sha256
    t=ctrl.prepare(episode=12)
    assert t["previous_train_probes_consumed"]==0
    assert t["head_only_objective_forwards"]==0
    scores=ctrl.forecast([direction]*8,context,episode=12)
    assert len(scores)==8
    assert all(x==scores[0] for x in scores)
    obs=ctrl.observe(direction,context,.0003,episode=12)
    assert obs["new_observations"]==1
    assert ctrl.head.parameter_sha256==original
    follow=ctrl.prepare(episode=13)
    assert follow["previous_train_probes_consumed"]==1
    assert 1<=follow["head_only_objective_forwards"]<=5
    assert 0<=follow["head_only_accepted_updates"]<=2
    assert follow["model_backwards"]==0
    assert follow["model_adapter_updates"]==0
    assert follow["uses_diagnostic_labels"] is False
    assert follow["uses_current_episode_labels"] is False
    ctrl.forecast([direction]*8,context,episode=13)
    ctrl.observe(direction,context,-.04,episode=13)
    again=ctrl.prepare(episode=14)
    assert again["previous_train_probes_consumed"]==1
    assert again["model_weight_promotion_permitted"] is False
    assert ctrl.total_head_accepted_updates<=4
    assert ctrl.head.update_norm<=.6+1e-9


def test_unmeasured_future_or_extra_feedback_rejected():
    ctrl,source=new_controller()
    context=source["events"][0]["context"]
    direction=[1.]+[0.]*7
    with pytest.raises(ValueError):
        ctrl.prepare(episode=13)
    with pytest.raises(ValueError):
        ctrl.observe(direction,context,.1,episode=12)
    ctrl.prepare(episode=12)
    with pytest.raises(ValueError):
        ctrl.prepare(episode=12)
    with pytest.raises(ValueError):
        ctrl.observe(direction,context,float("nan"),episode=12)
    with pytest.raises(ValueError):
        ctrl.forecast([direction]*7,context,episode=12)
    ctrl.observe(direction,context,.1,episode=12)
    with pytest.raises(ValueError):
        ctrl.observe(direction,context,.1,episode=12)
    with pytest.raises(ValueError):
        ctrl.forecast([direction]*8,context,episode=12)
    ctrl.prepare(episode=13)
    with pytest.raises(ValueError):
        ctrl.prepare(episode=14)


def test_adaptive_head_cannot_take_unregistered_seed():
    _,source=new_controller()
    for seed in (10,25,27,31):
        with pytest.raises(ValueError):
            OnlineSingleFeedbackHead(rebuild_training_head(source),seed=seed)


@pytest.mark.parametrize("seed",SEEDS)
def test_task_family_disjointness_and_equal_probes_from_new_seeds(seed):
    episodes=tasks(seed)
    assert [e["episode"] for e in episodes]==[12,13,14]
    keys=[x["input_key"] for row in episodes
          for label in ("training","diagnostic_same_family","diagnostic_transfer")
          for x in row[label]]
    assert len(keys)==21
    assert len(set(keys))==21
    for ep in (12,13,14):
        c=candidates(seed,ep)
        assert len(c)==8
        for i in range(0,8,2):
            assert c[i]["direction"]==[-v for v in c[i+1]["direction"]]
        assert policy_choice(seed,ep,[0.]*8,[0.]*8)==policy_choice(seed,ep,[0.]*8,[0.]*8)
    with pytest.raises(ValueError):
        candidates(seed,15)
