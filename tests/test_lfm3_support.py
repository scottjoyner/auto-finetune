"""LFM3 support helpers."""
from src.lfm3_support import is_lfm3_model, LFM3_MODEL_LOCAL_PATH


def test_is_lfm3_model_matches_name_and_path():
    assert is_lfm3_model("LiquidAI/LFM3-1.7B")
    assert is_lfm3_model("lfm3")
    assert is_lfm3_model("/media/scott/data/finetune-staging/models/LFM3-1.7B")
    assert not is_lfm3_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert not is_lfm3_model(None)
    assert not is_lfm3_model("")


def test_local_path_points_at_model_dir():
    assert "LFM3-1.7B" in LFM3_MODEL_LOCAL_PATH
