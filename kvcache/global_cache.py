"""GlobalKVCache — paged KV-cache for global-attention layers.

Global attention runs on full context (262k tokens). Full KV does not fit
on one GPU, so we use paged blocks with eviction.

Block size: 4096 tokens → 65 blocks total (262144 / 4096)
Hot blocks: gfx1201 (32GB primary)
Cold blocks: evicted to host RAM or gfx1150 (8GB secondary)
Quantization: int8 for cold blocks to halve memory.
"""
import torch
import time
from enum import Enum

class BlockStatus(Enum):
    HOT = "hot"        # cached on primary GPU (gfx1201)
    COLD = "cold"      # evicted, in host RAM or secondary GPU
    QUANTIZED = "quantized"  # int8 on secondary GPU or host

class GlobalKVCache:
    def __init__(self, num_layers=62, context_length=262144, block_size=4096,
                 hidden_dim=128, num_kv_heads=8, device_primary='cuda:0',
                 device_secondary='cuda:1'):
        self.context_length = context_length
        self.block_size = block_size
        self.num_blocks = context_length // block_size  # 65
        self.hidden_dim = hidden_dim
        self.num_kv_heads = num_kv_heads
        self.device_primary = device_primary
        self.device_secondary = device_secondary
        self.current_layer = 0

        # Block metadata: status, device, last_access
        self.blocks = {
            i: {
                'status': BlockStatus.COLD,
                'device': None,
                'last_access': 0.0,
                'data': None  # (K, V) tensors, lazy-loaded
            }
            for i in range(self.num_blocks)
        }

        # Pre-allocate hot block pool on primary GPU (32GB)
        # Each block: K = (block_size, num_kv_heads, head_dim), V same
        # head_dim = hidden_dim // num_kv_heads = 16
        self.head_dim = hidden_dim // num_kv_heads
        self.block_bytes = 2 * block_size * num_kv_heads * self.head_dim * 2  # K+V, fp16
        self.max_hot_blocks = 32 // (self.block_bytes / 1e9)  # ~32GB budget
        self.hot_pool = []

        # LRU tracking
        self.access_counter = 0

    def allocate_block(self, block_id: int, layer: int, device: str = 'primary'):
        """Allocate K/V for a block on specified device."""
        target_device = self.device_primary if device == 'primary' else self.device_secondary
        k = torch.zeros(self.block_size, self.num_kv_heads, self.head_dim, dtype=torch.float16, device=target_device)
        v = torch.zeros(self.block_size, self.num_kv_heads, self.head_dim, dtype=torch.float16, device=target_device)
        self.blocks[block_id]['data'] = (k, v)
        self.blocks[block_id]['status'] = BlockStatus.HOT if device == 'primary' else BlockStatus.COLD
        self.blocks[block_id]['device'] = target_device
        self.blocks[block_id]['last_access'] = time.time()
        self.blocks[block_id]['layer'] = layer
        if device == 'primary' and block_id not in self.hot_pool:
            self.hot_pool.append(block_id)

    def get_block(self, block_id: int, layer: int) -> tuple:
        """Retrieve K/V block, evicting/cold-loading as needed. LRU eviction."""
        self.access_counter += 1
        block = self.blocks[block_id]
        block['last_access'] = time.time()

        if block['status'] == BlockStatus.HOT and block['data'] is not None:
            return block['data']

        # Cold block: need to load
        if block['data'] is None:
            # Allocate fresh (from model KV or host)
            self.allocate_block(block_id, layer, 'primary' if len(self.hot_pool) < self.max_hot_blocks else 'secondary')
        elif block['status'] == BlockStatus.COLD:
            # Evict a cold block if needed, load this one to primary
            self._evict_lru()
            self.allocate_block(block_id, layer, 'primary')

        return self.blocks[block_id]['data']

    def _evict_lru(self):
        """Evict least-recently-used hot block to secondary/cold."""
        if not self.hot_pool:
            return
        # Find LRU hot block
        lru_block = min(self.hot_pool, key=lambda bid: self.blocks[bid]['last_access'])
        self.blocks[lru_block]['status'] = BlockStatus.COLD
        self.blocks[lru_block]['device'] = self.device_secondary
        self.hot_pool.remove(lru_block)

    def quantize_block(self, block_id: int):
        """Quantize a cold block to int8 to halve memory."""
        if self.blocks[block_id]['data'] is None:
            return
        k, v = self.blocks[block_id]['data']
        self.blocks[block_id]['data'] = (k.to(torch.int8), v.to(torch.int8))
        self.blocks[block_id]['status'] = BlockStatus.QUANTIZED

    def prefetch_blocks(self, block_ids: list, layer: int):
        """Prefetch blocks before attention trigger (async)."""
        for bid in block_ids:
            if self.blocks[bid]['status'] == BlockStatus.COLD or self.blocks[bid]['data'] is None:
                self.allocate_block(bid, layer, 'primary')

    def reset(self):
        for bid in self.blocks:
            self.blocks[bid]['status'] = BlockStatus.COLD
            self.blocks[bid]['data'] = None
            self.blocks[bid]['device'] = None
            self.blocks[bid]['last_access'] = 0.0
        self.hot_pool = []
        self.access_counter = 0

    @property
    def hot_memory_bytes(self) -> int:
        total = 0
        for bid in self.hot_pool:
            if self.blocks[bid]['data'] is not None:
                total += self.block_bytes
        return total

    def __repr__(self):
        return f"GlobalKVCache(blocks={self.num_blocks}, hot={len(self.hot_pool)}, mem={self.hot_memory_bytes/1e9:.2f}GB)"
