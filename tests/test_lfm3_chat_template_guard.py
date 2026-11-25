"""Native format integrity tests; synthetic tokens, no model weights or GPU."""
import hashlib
from pathlib import Path
import pytest
from src import lfm3_chat_template_guard as guard


class Tokenizer:
    bos_token = "<|startoftext|>"
    eos_token = "<|im_end|>"
    pad_token = "<|pad|>"
    bos_token_id = 124894
    eos_token_id = 124900
    pad_token_id = 124893

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, *args, **kwargs):
        return "synthetic"


def fixture(tmp_path, monkeypatch):
    template = "{{ bos_token }}<|im_start|>user<|im_end|>{{ tools }}"
    file = tmp_path / "chat_template.jinja"
    file.write_text(template, encoding="utf-8")
    monkeypatch.setattr(guard, "OFFICIAL_CHAT_TEMPLATE_SHA256",
                        hashlib.sha256(template.encode()).hexdigest())
    return template, file


def test_preserves_native_template_without_mutating_it(tmp_path, monkeypatch):
    template, _ = fixture(tmp_path, monkeypatch)
    tok = Tokenizer(template)
    assert guard.ensure_native_lfm3_chat_template(tok, tmp_path) is tok
    assert tok.chat_template == template


def test_wrong_native_template_rejected(tmp_path, monkeypatch):
    template, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(guard.LFM3ChatTemplateError, match="native_template_not_loaded"):
        guard.ensure_native_lfm3_chat_template(Tokenizer(template + "rewrite"), tmp_path)


def test_missing_native_template_rejected(tmp_path):
    with pytest.raises(guard.LFM3ChatTemplateError, match="native_template_missing"):
        guard.ensure_native_lfm3_chat_template(Tokenizer("anything"), tmp_path)


def test_tampered_native_template_rejected(tmp_path, monkeypatch):
    template, file = fixture(tmp_path, monkeypatch)
    file.write_text(template + "tampered")
    with pytest.raises(guard.LFM3ChatTemplateError, match="digest_mismatch"):
        guard.ensure_native_lfm3_chat_template(Tokenizer(template), tmp_path)


def test_symlink_native_template_rejected(tmp_path, monkeypatch):
    template, file = fixture(tmp_path, monkeypatch)
    target = tmp_path / "target"
    file.rename(target)
    file.symlink_to(target)
    with pytest.raises(guard.LFM3ChatTemplateError, match="linked_native_template"):
        guard.ensure_native_lfm3_chat_template(Tokenizer(template), tmp_path)


def test_wrong_eos_refused(tmp_path, monkeypatch):
    template, _ = fixture(tmp_path, monkeypatch)
    tok = Tokenizer(template)
    tok.eos_token = "<|end|>"
    with pytest.raises(guard.LFM3ChatTemplateError, match="special_token_mismatch:eos_token"):
        guard.ensure_native_lfm3_chat_template(tok, tmp_path)


def test_wrong_token_id_refused(tmp_path, monkeypatch):
    template, _ = fixture(tmp_path, monkeypatch)
    tok = Tokenizer(template)
    tok.pad_token_id = 100
    with pytest.raises(guard.LFM3ChatTemplateError, match="special_token_mismatch:pad_token"):
        guard.ensure_native_lfm3_chat_template(tok, tmp_path)


def test_no_renderer_refused(tmp_path, monkeypatch):
    template, _ = fixture(tmp_path, monkeypatch)
    tok = Tokenizer(template)
    tok.apply_chat_template = None
    with pytest.raises(guard.LFM3ChatTemplateError, match="missing_native_template_renderer"):
        guard.ensure_native_lfm3_chat_template(tok, tmp_path)


def test_trainer_wires_native_guard_without_overriding_template():
    source = (Path(__file__).parents[1] / "src" / "train.py").read_text()
    assert "ensure_native_lfm3_chat_template(tokenizer, lfm3_dir)" in source
    assert "tokenizer.chat_template =" not in source
    assert "pinned LFM3 tokenizer load failed; fallback refused" in source


def test_pinned_snapshot_includes_native_template_digest():
    from src.lfm3_pretrained_guard import OFFICIAL_LFM3_SHA256
    assert OFFICIAL_LFM3_SHA256["chat_template.jinja"] == (
        guard.OFFICIAL_CHAT_TEMPLATE_SHA256
    )
