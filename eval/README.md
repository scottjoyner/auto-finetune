# Option Evaluation System

Evaluates an array of options against desired outcomes with a self-improving reward model.

## Components
- OptionEvaluator: scores options against weighted criteria
- RewardModel: learns from feedback, self-updating weights
- FeedbackLoop: collects outcomes, retrains reward model
- Configurable to any decision space (KV-cache, ring topology, etc.)

## Usage
from eval.system import EvaluationSystem
system = EvaluationSystem(criteria, options)
scores = system.evaluate()
system.reward(feedback)  # self-improving
