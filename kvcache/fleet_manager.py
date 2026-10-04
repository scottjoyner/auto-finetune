"""FleetKVManager — multi-GPU KV-cache sharding and coherence.

Shards global KV-cache by layer type (not by tensor).
Coordinates delta and global cache across fleet nodes.
Handles coherence tracking (hot/cold block metadata).
"""
import torch
import time
from collections import defaultdict
from .delta_cache import DeltaKVCache
from .global_cache import GlobalKVCache, BlockStatus

class FleetKVManager:
    def __init__(self, num_gpus=2, context_length=262144, hidden_dim=128,
                 num_kv_heads=8, block_size=4096):
        self.num_gpus = num_gpus
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.block_size = block_size

        # Device map: primary (gfx1201 32GB), secondary (gfx1150 8GB)
        self.device_primary = 'cuda:0'
        self.device_secondary = 'cuda:1' if num_gpus > 1 else 'cuda:0'

        # Per-layer KV cache
        # Global layers (every 4th): GlobalKVCache
        # Delta layers: DeltaKVCache
        self.global_layers = [i for i in range(62) if i % 4 == 0]  # layers 0,4,8,...
        self.delta_layers = [i for i in range(62) if i % 4 != 0]
        self.global_caches = {}  # layer -> GlobalKVCache
        self.delta_caches = {}   # layer -> DeltaKVCache

        # Initialize caches
        for layer in self.global_layers:
            self.global_caches[layer] = GlobalKVCache(
                num_layers=62, context_length=context_length,
                block_size=block_size, hidden_dim=hidden_dim,
                num_kv_heads=num_kv_heads,
                device_primary=self.device_primary,
                device_secondary=self.device_secondary
            )
        for layer in self.delta_layers:
            self.delta_caches[layer] = DeltaKVCache(
                hidden_dim=hidden_dim, max_tokens=context_length,
                device=self.device_primary
            )

        # Coherence tracking (per block: last_access, importance score)
        self.coherence = defaultdict(lambda: {'last_access': 0, 'importance': 0.0})
        self.fleet_nodes = []

    def register_node(self, node_id: str, device: str):
        """Register a fleet node."""
        self.fleet_nodes.append({'id': node_id, 'device': device})

    def get_kv(self, layer: int, block_id: int) -> tuple:
        """Get K/V for a layer and block, routing to correct cache."""
        if layer in self.global_layers:
            return self.global_caches[layer].get_block(block_id, layer)
        elif layer in self.delta_layers:
            return self.delta_caches[layer].get_state(block_id * self.block_size,
                                                       (block_id + 1) * self.block_size)
        raise ValueError(f"Unknown layer {layer}")

    def update_delta(self, layer: int, position: int, hidden_delta: torch.Tensor):
        """Update delta cache for a new token."""
        if layer not in self.delta_layers:
            raise ValueError(f"Layer {layer} is not a delta layer")
        self.delta_caches[layer].update(hidden_delta, position)

    def update_coherence(self, block_id: int, importance: float):
        """Update coherence metadata for a block."""
        self.coherence[block_id]['last_access'] = time.time()
        self.coherence[block_id]['importance'] = importance

    def prefetch(self, block_ids: list, layer: int):
        """Prefetch blocks for upcoming attention."""
        if layer in self.global_layers:
            self.global_caches[layer].prefetch_blocks(block_ids, layer)

    def quantize_cold(self, layer: int, block_ids: list):
        """Quantize cold blocks to int8."""
        if layer in self.global_layers:
            for bid in block_ids:
                self.global_caches[layer].quantize_block(bid)

    def coherence_report(self) -> dict:
        """Generate coherence report for monitoring."""
        report = {
            'global_layers': len(self.global_layers),
            'delta_layers': len(self.delta_layers),
            'total_blocks': sum(len(gc.blocks) for gc in self.global_caches.values()),
            'hot_blocks': sum(len(gc.hot_pool) for gc in self.global_caches.values()),
            'hot_memory_gb': sum(gc.hot_memory_bytes for gc in self.global_caches.values()) / 1e9,
            'coherence_blocks': len(self.coherence),
            'fleet_nodes': len(self.fleet_nodes)
        }
        return report

    def reset_all(self):
        """Reset all caches."""
        for gc in self.global_caches.values():
            gc.reset()
        for dc in self.delta_caches.values():
            dc.reset()
        self.coherence.clear()

    def __repr__(self):
        report = self.coherence_report()
        return f"FleetKVManager(nodes={report['fleet_nodes']}, hot_mem={report['hot_memory_gb']:.1f}GB, hot_blocks={report['hot_blocks']})"
