"""Require the pinned LFM3 native tool-capable chat template before training.

No remote fetch, model allocation, template rewrite, or weight modification.
"""
from __future__ import annotations

import hashlib
import stat
from pathlib import Path

OFFICIAL_CHAT_TEMPLATE_SHA256 = (
    "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"
)
EXPECTED_SPECIAL = {
    "bos_token": ("<|startoftext|>", 124894),
    "eos_token": ("<|im_end|>", 124900),
    "pad_token": ("<|pad|>", 124893),
}
MAX_BYTES = 65536


class LFM3ChatTemplateError(ValueError):
    """Cannot safely format pinned LFM3 chat/tool training examples."""


def ensure_native_lfm3_chat_template(tokenizer, checkpoint_dir):
    """Inspect a local, already-pinned tokenizer and do not mutate it."""
    file = Path(checkpoint_dir) / "chat_template.jinja"
    try:
        if file.is_symlink():
            raise LFM3ChatTemplateError("linked_native_template")
        info = file.stat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise LFM3ChatTemplateError("invalid_native_template_file")
        blob = file.read_bytes()
    except OSError as exc:
        raise LFM3ChatTemplateError("native_template_missing_or_unreadable") from exc
    if hashlib.sha256(blob).hexdigest() != OFFICIAL_CHAT_TEMPLATE_SHA256:
        raise LFM3ChatTemplateError("native_template_digest_mismatch")
    try:
        expected = blob.decode("utf-8")
    except UnicodeError as exc:
        raise LFM3ChatTemplateError("invalid_native_template_encoding") from exc
    current = getattr(tokenizer, "chat_template", None)
    if type(current) is not str or current != expected:
        raise LFM3ChatTemplateError("native_template_not_loaded")
    for name, (token, token_id) in EXPECTED_SPECIAL.items():
        if (getattr(tokenizer, name, None) != token
                or getattr(tokenizer, name + "_id", None) != token_id):
            raise LFM3ChatTemplateError("special_token_mismatch:" + name)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise LFM3ChatTemplateError("missing_native_template_renderer")
    # Explicitly do not set tokenizer.chat_template: this is an integrity
    # assertion, not a replacement chat serializer.
    return tokenizer
