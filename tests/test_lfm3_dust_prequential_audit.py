"""CPU-only contracts for unbiased prequential LFM3 direction forecasts."""
import copy
import math

import pytest
from src.lfm3_dust_prequential_audit import (
    CANDIDATES, EPISODES, POLICIES, SIGMA, _metrics, validate,
)


def _episodes():
    rows=[]
    for episode in range(EPISODES):
        cases=[]
        for index in range(2 * CANDIDATES):
            gain=(index - 3.5) / 100
            cases.append({"measured_gain":gain,"predictions":{
                "zero":0., "historical_mean":0.,
                "direction":gain, "activation":gain/2,
                "calibrated":gain/4}})
        rows.append({"episode":episode,"candidates":cases})
    return rows


def test_metrics_are_prospective_and_skip_first_episode():
    scores=_metrics(_episodes())
    assert set(scores)==set(POLICIES)
    assert all(x["cases"]==(EPISODES-1)*2*CANDIDATES for x in scores.values())
    assert scores["direction"]["mse"] == pytest.approx(0)
    assert scores["direction"]["mean_top1_regret"] == pytest.approx(0)
    assert scores["activation"]["mse"] > 0
    assert scores["zero"]["mse"] > scores["activation"]["mse"]


def test_cold_start_is_not_scored():
    rows=_episodes()
    original=_metrics(rows)
    for x in rows[0]["candidates"]:
        x["predictions"]["direction"]=1e6
        x["measured_gain"]=-1e6
    assert _metrics(rows)==original


def test_incomplete_episode_history_denied():
    rows=_episodes()[:-1]
    with pytest.raises(ValueError,match="incomplete"):
        _metrics(rows)


def test_nonfinite_forecast_denied():
    rows=_episodes()
    rows[1]["candidates"][0]["predictions"]["calibrated"]=float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _metrics(rows)


def test_bounds_refuse_fewer_episodes_or_more_directions():
    validate(seed=1,episodes=8,candidates=4,sigma=SIGMA)
    for setting in (
        {"seed":True,"episodes":8,"candidates":4,"sigma":.05},
        {"seed":1,"episodes":3,"candidates":4,"sigma":.05},
        {"seed":1,"episodes":8,"candidates":5,"sigma":.05},
        {"seed":1,"episodes":8,"candidates":4,"sigma":.7},
        {"seed":1,"episodes":8,"candidates":4,"sigma":float("nan")},
    ):
        with pytest.raises(ValueError):
            validate(**setting)


def test_experiment_must_be_offline_cpu_before_model_imports(monkeypatch):
    from src.lfm3_dust_prequential_audit import run
    monkeypatch.delenv("HF_HUB_OFFLINE",raising=False)
    with pytest.raises(RuntimeError,match="offline_cpu_isolation"):
        run(seed=1)
