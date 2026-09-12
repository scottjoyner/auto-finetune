"""Criteria — defines outcome dimensions and weights for evaluation."""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Criteria:
    name: str
    dimensions: Dict[str, float]  # dimension_name -> weight
    constraints: Dict[str, float] = field(default_factory=dict)  # min/max bounds

    def validate(self, option: Dict) -> bool:
        """Check if option satisfies constraints."""
        for key, (lo, hi) in self.constraints.items():
            if key in option and not (lo <= option[key] <= hi):
                return False
        return True

    def score(self, option: Dict) -> float:
        """Weighted score for an option (0-1 normalized)."""
        total = 0.0
        for dim, weight in self.dimensions.items():
            val = option.get(dim, 0.0)
            total += weight * min(max(val, 0.0), 1.0)  # clamp to [0,1]
        return total

# Pre-built criteria templates
CRITERIA_KV_CACHE = Criteria(
    name="kv_cache_config",
    dimensions={
        'memory_fit': 0.3,       # fits in VRAM
        'throughput': 0.25,      # tokens/sec
        'latency': 0.2,          # first-token latency
        'coherence': 0.15,       # attention accuracy
        'fault_tolerance': 0.1   # ring-node resilience
    },
    constraints={
        'memory_fit': (0.0, 1.0),
        'throughput': (0.0, 1000.0),
        'latency': (0.0, 1000.0),
        'coherence': (0.0, 1.0)
    }
)

CRITERIA_RING_TOPOLOGY = Criteria(
    name="ring_topology",
    dimensions={
        'latency': 0.3,
        'bandwidth_util': 0.25,
        'fault_tolerance': 0.25,
        'scalability': 0.2
    },
    constraints={
        'latency': (0.0, 100.0),  # ms
        'bandwidth_util': (0.0, 1.0),
        'fault_tolerance': (0.0, 1.0),
        'scalability': (1, 64)
    }
)
