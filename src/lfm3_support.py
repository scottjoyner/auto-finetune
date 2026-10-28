"""LFM3 model support for auto-finetune.

LiquidAI LFM3-1.7B is a hybrid architecture model with:
  - Conv layers (no RoPE)
  - Full attention layers (RoPE with rope_theta=1M, partial_rotary_factor=0.5)
  - Sliding attention layers (RoPE with rope_theta=10K)

Transformers 5.x has a bug with nested rope_parameters for LFM3.
This module provides patches to make LFM3 work with the training pipeline.
"""
from __future__ import annotations

import os
import sys
from typing import Any

LFM3_MODEL_NAME = "LiquidAI/LFM3-1.7B"
LFM3_MODEL_LOCAL_PATH = "/media/scott/data/finetune-staging/models/LFM3-1.7B"
LFM3_LOCAL_PATH = "/media/scott/data/finetune-staging/models/LFM3-1.7B"


def patch_transformers_for_lfm3() -> None:
    """Apply monkey-patches to transformers to support LFM3's nested rope_parameters.

    LFM3 uses per-layer-type RoPE parameters (full_attention, sliding_attention,
    conv). Transformers 5.x's convert_rope_params_to_dict adds rope_theta at the
    top level of rope_parameters, which breaks validation for nested dicts.
    """
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
    except ImportError:
        return

    # Patch convert_rope_params_to_dict to not add rope_theta at top level
    # when rope_parameters is a nested dict (per-layer-type)
    original_convert = RotaryEmbeddingConfigMixin.convert_rope_params_to_dict

    def patched_convert(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        layer_types = getattr(self, "_rope_type_labels",
                              getattr(self, "layer_types", None))
        is_nested = layer_types is not None and any(
            k in self.rope_parameters for k in layer_types if k
        )

        if not is_nested:
            rope_theta = kwargs.pop("rope_theta",
                                    getattr(self, "rope_theta", self.default_theta))
            self.rope_parameters.setdefault("rope_theta", rope_theta)
            partial_rotary_factor = kwargs.get(
                "partial_rotary_factor",
                getattr(self, "partial_rotary_factor", None),
            )
            if partial_rotary_factor is not None:
                self.rope_parameters.setdefault(
                    "partial_rotary_factor", partial_rotary_factor
                )
                self.ignore_keys_at_rope_validation = set(
                    self.ignore_keys_at_rope_validation or []
                ) | {"partial_rotary_factor"}

        self.standardize_rope_params()
        return kwargs

    RotaryEmbeddingConfigMixin.convert_rope_params_to_dict = patched_convert

    # Patch standardize_rope_params to handle missing layer types
    original_std = RotaryEmbeddingConfigMixin.standardize_rope_params

    def patched_std(self):
        try:
            original_std(self)
        except KeyError as e:
            missing_key = str(e).strip("'")
            layer_types = getattr(self, "_rope_type_labels",
                                  getattr(self, "layer_types", None))
            if layer_types is not None and missing_key in layer_types:
                self.rope_parameters[missing_key] = None
                original_std(self)

    RotaryEmbeddingConfigMixin.standardize_rope_params = patched_std

    # Patch validate_rope to skip non-dict values (like rope_theta float at top level)
    original_validate = RotaryEmbeddingConfigMixin.validate_rope

    def patched_validate(self):
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return
        layer_types = getattr(self, "layer_types", None)
        if layer_types is not None and not set(
            rope_parameters_dict.keys()
        ).isdisjoint(layer_types):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}
        for key, rp in rope_parameters_dict.items():
            if rp is None:
                continue
            if not isinstance(rp, dict):
                continue
            rope_type = rp.get("rope_type", rp.get("type", "default"))
            rp["rope_type"] = rope_type
            validation_fn = getattr(
                self, f"_validate_{rope_type}_rope_parameters", None
            )
            if validation_fn is not None:
                validation_fn(
                    rp, ignore_keys=self.ignore_keys_at_rope_validation
                )

    RotaryEmbeddingConfigMixin.validate_rope = patched_validate

def is_lfm3_model(model_name: str) -> bool:
    """Check if the model name refers to an LFM3 model."""
    if not model_name:
        return False
    return "lfm3" in model_name.lower()


def get_lfm3_config_overrides() -> dict[str, Any]:
    """Get config overrides for LFM3 models."""
    return {
        "max_seq_length": 4096,  # LFM3 supports up to 128K, but 4K is practical for 15GB VRAM
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "load_in_4bit": False,  # LFM3-1.7B fits in bf16 on 15GB VRAM
        "use_gradient_checkpointing": True,
    }


def patch_and_import() -> None:
    """Patch transformers and import LFM3 modules."""
    patch_transformers_for_lfm3()

    # Ensure custom modeling code is importable
    model_dir = LFM3_LOCAL_PATH
    if os.path.isdir(model_dir):
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)


def load_lfm3_model_and_tokenizer(
    model_name: str = LFM3_MODEL_NAME,
    local_path: str = LFM3_LOCAL_PATH,
) -> tuple[Any, Any]:
    """Load LFM3 model and tokenizer with patches applied.

    Returns (model, tokenizer).
    """
    patch_and_import()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Try local path first, fall back to HF hub
    model_path = local_path if os.path.isdir(local_path) else model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model, tokenizer
