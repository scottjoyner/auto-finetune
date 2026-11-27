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


def test_patch_if_lfm3_returns_false_for_non_lfm3():
    from src.lfm3_support import patch_if_lfm3
    assert patch_if_lfm3("Qwen/Qwen2.5-0.5B-Instruct") is False


def test_load_lfm3_chat_template_returns_false_for_non_lfm3():
    from src.lfm3_support import load_lfm3_chat_template
    class Dummy:
        pass
    d = Dummy()
    assert load_lfm3_chat_template(d, "Qwen/Qwen2.5-0.5B-Instruct") is False


def test_patch_if_lfm3_true_for_lfm3_name():
    from src.lfm3_support import patch_if_lfm3
    # Should not raise; returns True for LFM3 names
    assert patch_if_lfm3("LiquidAI/LFM3-1.7B") is True


def test_load_lfm3_chat_template_applies_for_lfm3_path():
    from src.lfm3_support import load_lfm3_chat_template, LFM3_MODEL_LOCAL_PATH
    import os
    if not os.path.isdir(LFM3_MODEL_LOCAL_PATH):
        import pytest
        pytest.skip("LFM3 model dir missing")
    class Dummy:
        chat_template = None
    d = Dummy()
    assert load_lfm3_chat_template(d, LFM3_MODEL_LOCAL_PATH) is True
    assert d.chat_template is not None


def test_load_lfm3_chat_template_fails_closed_when_missing(monkeypatch):
    """No inline fallback: train and eval must not silently disagree."""
    import pytest
    import src.lfm3_support as support

    monkeypatch.setattr(support, "LFM3_MODEL_LOCAL_PATH", "/nonexistent/lfm3")

    class Dummy:
        chat_template = None

    with pytest.raises(FileNotFoundError):
        support.load_lfm3_chat_template(Dummy(), "/also-missing/LFM3-1.7B")
