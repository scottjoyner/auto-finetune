import json
from pathlib import Path
import pytest

from experiments.dust.summarize_tail_runs import summarize


def manifest(seed, train_delta, held_delta, sec):
    dataset={
        "train_source":{"sha256":"t"},
        "eval_source":{"sha256":"e"},
        "train_selected_pair_sha256":["a","b"],
        "eval_selected_pair_sha256":["x","y"],
        "train_rows":2,"eval_rows":2,
        "train_max_seq_tokens":128,"eval_max_seq_tokens":512,
    }
    return {
        "schema":"auto-finetune.dust-k2-tail-train.v1",
        "model_weights_sha256":"m","model_config_sha256":"c",
        "rank":4,"steps":4,"population":1024,"sigma":.25,
        "learning_rate":.1,"direction_batch":4,"seed":seed,
        "dataset":dataset,
        "train_ce_delta":train_delta,"heldout_ce_delta":held_delta,
        "training_seconds":sec,"total_seconds":sec+2,
        "prefix_cache_seconds":1,
        "reference":{
            "serial_structured":{
                "train_ce_delta":-.01,
                "heldout_ce_delta":.02,
                "elapsed_seconds":100,
            },
            "backprop":{
                "train_ce_delta":-.02,
                "heldout_ce_delta":.01,
                "elapsed_seconds":5,
            },
        },
    }


def write(path,obj):
    path.write_text(json.dumps(obj))
    return path


def test_tail_summary_checks_comparability_and_aggregates(tmp_path):
    a=write(tmp_path/"a.json",manifest(7,-.1,.01,20))
    b=write(tmp_path/"b.json",manifest(42,.1,-.02,30))
    out=summarize([a,b])
    assert out["seed_count"]==2
    assert out["tail"]["train_improved_seeds"]==1
    assert out["tail"]["heldout_improved_seeds"]==1
    assert out["tail"]["training_seconds"]["mean"]==pytest.approx(25)
    dup=write(tmp_path/"dup.json",manifest(7,.1,.1,10))
    with pytest.raises(ValueError,match="Duplicate"):
        summarize([a,dup])
    bad=manifest(1337,.1,.1,10)
    bad["direction_batch"]=8
    bp=write(tmp_path/"bad.json",bad)
    with pytest.raises(ValueError,match="not comparable"):
        summarize([a,bp])
