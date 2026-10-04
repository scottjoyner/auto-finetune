"""DeltaKVCache — running delta accumulator for local delta-attention layers.

Delta attention maintains a running state delta (not full K/V matrices).
Update is O(1) per new token: accumulate hidden-state delta.
Size is tokens × hidden_dim × 2 (fp16) — tiny (~0.1-1GB).
"""
import torch
import time

class DeltaKVCache:
    def __init__(self, hidden_dim=128, max_tokens=262144, device='cuda:0'):
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.device = device
        # Running delta accumulator (not full K/V)
        self.delta_state = torch.zeros(max_tokens, hidden_dim * 2, dtype=torch.float16, device=device)
        self.token_count = 0
        self.last_update = time.time()

    def update(self, hidden_delta: torch.Tensor, position: int):
        """Accumulate delta state for a new token. O(1) update."""
        if position >= self.max_tokens:
            raise ValueError(f"Position {position} exceeds max_tokens {self.max_tokens}")
        self.delta_state[position] = hidden_delta.to(torch.float16)
        self.token_count = max(self.token_count, position + 1)
        self.last_update = time.time()

    def get_state(self, start: int, end: int) -> torch.Tensor:
        """Retrieve delta state slice [start:end]."""
        return self.delta_state[start:end]

    def reset(self):
        self.token_count = 0
        self.delta_state.zero_()
        self.last_update = time.time()

    @property
    def memory_bytes(self) -> int:
        return self.delta_state.numel() * self.delta_state.element_size()

    def __repr__(self):
        return f"DeltaKVCache(tokens={self.token_count}, dim={self.hidden_dim}, mem={self.memory_bytes/1e9:.2f}GB)"
