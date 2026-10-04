"""RewardModel — self-improving reward model with feedback learning.

Maintains weights over outcome dimensions. Updates via gradient step
on feedback (positive/negative outcomes). Designed to be a learned
reward function that improves over time.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from .criteria import Criteria

class RewardModel(nn.Module):
    def __init__(self, criteria: Criteria, hidden_dim=64):
        super().__init__()
        self.criteria = criteria
        self.dim_names = list(criteria.dimensions.keys())
        self.dim_count = len(self.dim_names)

        # Learnable weights (initialized from criteria weights)
        self.weights = nn.Parameter(
            torch.tensor([criteria.dimensions[d] for d in self.dim_names], dtype=torch.float32)
        )
        self.hidden_dim = hidden_dim

        # MLP for non-linear reward scoring
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam([self.weights, *self.mlp.parameters()], lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Training history
        self.training_history: List[Dict] = []

    def forward(self, option_vector: torch.Tensor) -> torch.Tensor:
        """Compute reward score (0-1) for an option vector."""
        return self.mlp(option_vector)

    def reward(self, option: Dict) -> float:
        """Compute scalar reward for an option dict."""
        vector = torch.tensor([option.get(d, 0.0) for d in self.dim_names], dtype=torch.float32)
        with torch.no_grad():
            score = self.mlp(vector).item()
        return score

    def update(self, option: Dict, feedback: float):
        """
        Update reward model based on feedback.
        feedback: float in [0,1] (1=positive, 0=negative)
        """
        vector = torch.tensor([option.get(d, 0.0) for d in self.dim_names], dtype=torch.float32)
        target = torch.tensor([feedback], dtype=torch.float32)

        self.optimizer.zero_grad()
        pred = self.mlp(vector)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()

        self.training_history.append({
            'option': option,
            'feedback': feedback,
            'loss': loss.item(),
            'weights': self.weights.detach().numpy().tolist()
        })
        return loss.item()

    def get_weights(self) -> Dict[str, float]:
        """Current learned weights per dimension."""
        return {d: float(w) for d, w in zip(self.dim_names, self.weights.detach().numpy())}

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'weights': self.weights.data,
            'mlp': self.mlp.state_dict(),
            'history': self.training_history,
            'criteria': self.criteria
        }, path)

    def load(self, path: str):
        """Load model state."""
        ckpt = torch.load(path)
        self.weights.data = ckpt['weights']
        self.mlp.load_state_dict(ckpt['mlp'])
        self.training_history = ckpt['history']
        self.criteria = ckpt['criteria']
