"""LFM3 support helpers."""
from src.lfm3_support import (
    get_lfm3_config_overrides,
    is_lfm3_model,
    LFM3_MODEL_LOCAL_PATH,
)


def test_is_lfm3_model_matches_name_and_path():
    assert is_lfm3_model("LiquidAI/LFM3-1.7B")
    assert is_lfm3_model("lfm3")
    assert is_lfm3_model("/media/scott/data/finetune-staging/models/LFM3-1.7B")
    assert not is_lfm3_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert not is_lfm3_model(None)
    assert not is_lfm3_model("")


def test_local_path_points_at_model_dir():
    assert "LFM3-1.7B" in LFM3_MODEL_LOCAL_PATH


def test_lfm3_config_overrides_are_pinned():
    assert get_lfm3_config_overrides() == {
        "max_seq_length": 4096,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "load_in_4bit": False,
        "use_gradient_checkpointing": True,
    }
