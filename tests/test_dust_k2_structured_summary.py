import json
from pathlib import Path
import pytest
from experiments.dust.summarize_structured_runs import summarize

def manifest(seed, held_delta, train_delta, a=.9, b=.8, secs=10):
    return {
        "schema":"auto-finetune.dust-k2-structured-train.v1",
        "model_weights_sha256":"m","model_config_sha256":"c","rank":4,
        "steps":4,"population":1024,"sigma":.25,"learning_rate":.1,
        "seed":seed,
        "dataset":{
            "train_source":{"sha256":"t"},"eval_source":{"sha256":"e"},
            "train_selected_pair_sha256":["a","b"],
            "eval_selected_pair_sha256":["x","y"],
            "train_rows":2,"eval_rows":2,
            "train_max_seq_tokens":128,"eval_max_seq_tokens":512,
        },
        "backprop":{"train_ce_delta":-0.1,"heldout_ce_delta":0.01,"elapsed_seconds":1},
        "structured":{"train_ce_delta":train_delta,"heldout_ce_delta":held_delta,"elapsed_seconds":secs},
        "adapter_update_alignment":{"a":{"cosine":a},"b":{"cosine":b}},
    }

def write(p,obj): p.write_text(json.dumps(obj)); return p

def test_summary_requires_comparable_unique_seeds(tmp_path):
    a=write(tmp_path/"a.json",manifest(7,-.01,-.1))
    b=write(tmp_path/"b.json",manifest(42,.02,-.2,a=.8,b=.7,secs=12))
    r=summarize([a,b])
    assert r["seed_count"]==2
    assert r["structured"]["heldout_improved_seeds"]==1
    assert r["adapter_update_alignment"]["a_cosine_min"]==pytest.approx(.8)
    dup=write(tmp_path/"dup.json",manifest(7,.01,-.2))
    with pytest.raises(ValueError,match="Duplicate"):
        summarize([a,dup])
    bad=manifest(99,.01,-.2); bad["population"]=512
    bp=write(tmp_path/"bad.json",bad)
    with pytest.raises(ValueError,match="not comparable"):
        summarize([a,bp])
