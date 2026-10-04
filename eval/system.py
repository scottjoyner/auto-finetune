"""EvaluationSystem — scores options, applies feedback, self-improves.

Main interface: evaluate options against criteria, collect outcomes,
update reward model, rank options.

Applied to KV-cache configs, ring topology, or any decision space.
"""
from typing import Dict, List, Optional
from .criteria import Criteria
from .reward_model import RewardModel
from .feedback_loop import FeedbackLoop

class EvaluationSystem:
    def __init__(self, criteria: Criteria, options: Optional[List[Dict]] = None):
        self.criteria = criteria
        self.reward_model = RewardModel(criteria)
        self.feedback_loop = FeedbackLoop(self.reward_model)
        self.options = options or []
        self.scores: Dict[str, float] = {}

    def add_option(self, option: Dict):
        """Add an option to evaluate."""
        self.options.append(option)

    def evaluate(self) -> Dict[str, float]:
        """Score all options against criteria + learned reward."""
        self.scores = {}
        for opt in self.options:
            # Base criteria score (weighted sum)
            base_score = self.criteria.score(opt)
            # Learned reward model score
            learned_score = self.reward_model.reward(opt)
            # Combined (50/50 blend — evolves as reward model learns)
            combined = 0.5 * base_score + 0.5 * learned_score
            self.scores[str(opt)] = combined
        return self.scores

    def rank(self) -> List[str]:
        """Return options ranked by combined score."""
        return sorted(self.scores, key=self.scores.get, reverse=True)

    def apply_feedback(self, option_key: str, outcome: Dict):
        """Apply outcome feedback to reward model."""
        option = self._get_option(option_key)
        if option is None:
            return None
        reward = self.feedback_loop.feedback(option, outcome)
        return reward

    def auto_evaluate(self, n_iterations=5):
        """
        Run automated evaluation loop with feedback.
        Evaluates options, simulates outcomes, updates reward model.
        """
        results = []
        for i in range(n_iterations):
            self.evaluate()
            ranked = self.rank()
            best = ranked[0] if ranked else None
            if best:
                # Simulate outcome for best option
                opt = self._get_option(best)
                outcome = self._simulate_outcome(opt)
                self.apply_feedback(best, outcome)
                results.append({
                    'iteration': i,
                    'best': best,
                    'reward': self.scores[best]
                })
        return results

    def _get_option(self, key: str) -> Optional[Dict]:
        for opt in self.options:
            if str(opt) == key:
                return opt
        return None

    def _simulate_outcome(self, option: Dict) -> Dict:
        """Simulate outcome for demo/testing. Real system uses actual metrics."""
        return {
            'memory_fit': option.get('memory_fit', 0.5),
            'throughput': option.get('throughput', 0.5),
            'latency': option.get('latency', 0.5),
            'coherence': option.get('coherence', 0.5)
        }

    def report(self) -> Dict:
        """Full evaluation system report."""
        return {
            'criteria': self.criteria.name,
            'n_options': len(self.options),
            'scores': self.scores,
            'ranked': self.rank(),
            'feedback': self.feedback_loop.report(),
            'best_option': self.rank()[0] if self.rank() else None
        }
