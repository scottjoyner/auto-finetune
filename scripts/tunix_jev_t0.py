"""No-dependency Tunix feasibility probe and synthetic Jev policy environment.

T0 only: not a Tunix integration, not gradient training, not a live agent.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from importlib import metadata


def probe_dependencies():
    result = {}
    for package in ("jax", "tunix"):
        present = importlib.util.find_spec(package) is not None
        version = None
        if present:
            for dist in ([package] if package == "jax" else ["google-tunix", "tunix"]):
                try:
                    version = metadata.version(dist)
                    break
                except metadata.PackageNotFoundError:
                    pass
        result[package] = {"module_found": present, "distribution_version": version}
    return result


class SyntheticProbeUpdate:
    """Two-step environment: observe a synthetic probe, choose a fake update."""

    def __init__(self, task_id="K2-synthetic-tail-1"):
        self.task_id = task_id
        self.phase = "probe"
        self.steps = 0
        seed = hashlib.sha256(task_id.encode("ascii")).digest()
        self._private_gradient = tuple((seed[i] - 127) / 127.0 for i in range(3))
        self.observation = {"task_id": task_id, "architecture": "synthetic-low-rank",
                            "ranks": [4, 8, 16], "action_schema": "jev.t0.v1"}

    def step(self, action):
        if self.phase == "probe":
            if type(action) is not dict or set(action) != {"op", "coordinate"} or action["op"] != "probe" or type(action["coordinate"]) is not int or action["coordinate"] not in range(3):
                raise ValueError("invalid_probe")
            index = action["coordinate"]
            self.steps += 1
            self.phase = "update"
            return {"probe_coordinate": index, "probe_value": self._private_gradient[index],
                    "stage": "update", "done": False}
        if self.phase != "update":
            raise ValueError("episode_finished")

        if action == {"op": "abstain"}:
            reward, direction = 0.0, None
        else:
            if (type(action) is not dict or set(action) != {"op", "coordinate", "sign"}
                    or action["op"] != "update"
                    or type(action["coordinate"]) is not int
                    or action["coordinate"] not in range(3)
                    or type(action["sign"]) is not int
                    or action["sign"] not in (-1, 1)):
                raise ValueError("invalid_update")
            direction = (action["coordinate"], action["sign"])
            reward = action["sign"] * self._private_gradient[action["coordinate"]] - 0.1
        self.phase = "done"
        return {"stage": "done", "done": True, "reward": round(reward, 8),
                "direction": direction, "hosted_calls": 0, "provider_tokens": 0,
                "production_dispatch_authorized": False}


def run():
    outcomes = []
    for task in ("K2-synthetic-tail-1", "K2-synthetic-tail-2"):
        env = SyntheticProbeUpdate(task)
        probe = env.step({"op": "probe", "coordinate": 0})
        sign = 1 if probe["probe_value"] >= 0 else -1
        result = env.step({"op": "update", "coordinate": 0, "sign": sign})
        outcomes.append({"task": task, "reward": result["reward"],
                         "provider_tokens": result["provider_tokens"],
                         "production_dispatch_authorized": result["production_dispatch_authorized"]})
    return {"schema": "jev.tunix-t0.v1", "dependencies": probe_dependencies(),
            "episodes": outcomes, "Tunix_agent_training_executed": False,
            "real_model_weights_read": False, "hosted_calls": 0,
            "status": "OFFLINE_ENVIRONMENT_ONLY"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
