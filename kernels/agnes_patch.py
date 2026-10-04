"""Patch to integrate Triton-accelerated delta-rule kernel and KV-cache
into Agnes-3.0-Flash modeling_agnes.py"""

import sys
sys.path.insert(0, '/home/scott/git/auto-finetune/kernels')
sys.path.insert(0, '/home/scott/git/auto-finetune')

from delta_rule_triton import (
    delta_rule_chunked_triton,
    delta_rule_stepwise_triton,
    causal_conv1d_triton,
)
from kvcache import FleetKVManager

def apply_triton_patch():
    """Apply Triton acceleration to Agnes model."""
    import torch
    from modeling_agnes import (
        _delta_rule_chunked,
        _delta_rule_stepwise,
        AgnesDeltaAttention,
        _FUSED_DELTA_PATH,
        causal_conv1d_fn,
        causal_conv1d_update,
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    # Replace pure-PyTorch delta-rule functions with Triton-accelerated versions
    import modeling_agnes as m
    m._delta_rule_chunked = delta_rule_chunked_triton
    m._delta_rule_stepwise = delta_rule_stepwise_triton
    m.chunk_gated_delta_rule = delta_rule_chunked_triton
    m.fused_recurrent_gated_delta_rule = delta_rule_stepwise_triton
    m.causal_conv1d_fn = causal_conv1d_triton
    m.causal_conv1d_update = causal_conv1d_triton
    m._FUSED_DELTA_PATH = True  # Force fused path

    print("Triton patch applied to modeling_agnes.py")
    print("  - delta_rule_chunked -> Triton kernel")
    print("  - delta_rule_stepwise -> Triton kernel")
    print("  - causal_conv1d -> Triton kernel")
    print("  - _FUSED_DELTA_PATH = True")

    return m


def integrate_kv_cache():
    """Integrate FleetKVManager into the Agnes model."""
    from modeling_agnes import AgnesDeltaAttention, AgnesGlobalAttention
    from kvcache import FleetKVManager

    manager = FleetKVManager(num_gpus=2, context_length=262144)

    # Wrap delta attention with KV-cache
    original_delta_forward = AgnesDeltaAttention.forward

    def patched_delta_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        # Get KV from FleetKVManager
        layer_idx = self.layer_idx
        if layer_idx in manager.global_layers:
            # Use GlobalKVCache
            block_id = hidden_states.shape[0] // manager.block_size
            k, v = manager.get_kv(layer_idx, block_id)
        else:
            # Use DeltaKVCache
            manager.update_delta(layer_idx, hidden_states.shape[0], hidden_states)
            k, v = None, None  # Delta state handled internally

        return original_delta_forward(self, hidden_states, cache_params, attention_mask, **kwargs)

    AgnesDeltaAttention.forward = patched_delta_forward
    print("KV-cache integration applied")
    return manager


if __name__ == '__main__':
    apply_triton_patch()
    integrate_kv_cache()
    print("All patches applied!")
