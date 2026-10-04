"""FeedbackLoop — collects outcomes and updates reward model.

Implements a feedback loop that:
1. Collects outcome data from evaluated options
2. Computes reward signal (positive/negative)
3. Updates reward model
4. Tracks improvement over time
"""
import time
from typing import Dict, List, Optional
from .reward_model import RewardModel
from .criteria import Criteria

class FeedbackLoop:
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self.outcomes: List[Dict] = []
        self.iteration = 0
        self.improvement_history: List[float] = []

    def record(self, option: Dict, outcome: Dict, reward: float):
        """Record an option's outcome and reward signal."""
        self.outcomes.append({
            'iteration': self.iteration,
            'option': option,
            'outcome': outcome,
            'reward': reward,
            'timestamp': time.time()
        })
        self.iteration += 1

    def feedback(self, option: Dict, outcome: Dict) -> float:
        """
        Compute reward from outcome and update model.
        Returns computed reward (0-1).
        """
        reward = self._compute_reward(outcome)
        self.reward_model.update(option, reward)
        self.record(option, outcome, reward)
        return reward

    def _compute_reward(self, outcome: Dict) -> float:
        """Derive reward signal from outcome dict."""
        # Default: reward = weighted sum of positive outcome metrics
        reward = 0.0
        for key, value in outcome.items():
            # Normalize value to [0,1] (higher = better)
            normalized = min(max(float(value) / 100.0 if isinstance(value, (int, float)) else 0.5, 0.0), 1.0)
            reward += normalized
        return min(reward / max(len(outcome), 1), 1.0)

    def get_improvement(self) -> float:
        """Average reward improvement over last N iterations."""
        if len(self.outcomes) < 2:
            return 0.0
        recent = self.outcomes[-10:]  # last 10
        rewards = [o['reward'] for o in recent]
        return (rewards[-1] - rewards[0]) if len(rewards) >= 2 else 0.0

    def best_option(self, options: List[Dict]) -> Dict:
        """Return highest-reward option from a list."""
        scored = [(self.reward_model.reward(opt), opt) for opt in options]
        return max(scored, key=lambda x: x[0])[1]

    def report(self) -> Dict:
        """Feedback loop status report."""
        return {
            'iterations': self.iteration,
            'outcomes': len(self.outcomes),
            'improvement': self.get_improvement(),
            'weights': self.reward_model.get_weights(),
            'recent_rewards': [o['reward'] for o in self.outcomes[-5:]]
        }
