"""PACE v2 entropy scheduling infrastructure, not wired into legacy training."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict

from .pace_v2_profiles import PaceV2EntropyConfig


def pace_v2_entropy_coefficient(
    iteration: int,
    config: PaceV2EntropyConfig,
) -> float:
    """Evaluate PACE v2 Eq. (21)-(22) for a fully specified schedule."""

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if config.initial < config.final or config.final < 0:
        raise ValueError("entropy coefficients must satisfy initial >= final >= 0")
    if config.turnover_iteration < 0:
        raise ValueError("entropy turnover must be non-negative")
    if config.slope_eta is None:
        raise RuntimeError("entropy slope eta is unresolved")
    if not math.isfinite(config.slope_eta) or config.slope_eta <= 0:
        raise ValueError("entropy slope eta must be finite and positive")

    transition = 0.5 - 0.5 * math.tanh(
        config.slope_eta * (iteration - config.turnover_iteration)
    )
    return config.final + transition * (config.initial - config.final)


class PaceV2EntropyScheduler:
    """Apply and checkpoint a specified entropy schedule against a PPO object."""

    def __init__(self, config: PaceV2EntropyConfig):
        self.config = config
        self.last_iteration = -1

    def coefficient(self, iteration: int) -> float:
        return pace_v2_entropy_coefficient(iteration, self.config)

    def apply(self, algorithm: Any, iteration: int) -> float:
        if not hasattr(algorithm, "entropy_coef"):
            raise TypeError("algorithm must expose an entropy_coef attribute")
        coefficient = self.coefficient(iteration)
        algorithm.entropy_coef = coefficient
        self.last_iteration = int(iteration)
        return coefficient

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": "pace_v2_entropy_scheduler.v1",
            "config": asdict(self.config),
            "last_iteration": self.last_iteration,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema") != "pace_v2_entropy_scheduler.v1":
            raise RuntimeError("unsupported entropy scheduler state")
        if state.get("config") != asdict(self.config):
            raise RuntimeError("checkpoint entropy schedule differs from runtime config")
        last_iteration = state.get("last_iteration")
        if not isinstance(last_iteration, int) or last_iteration < -1:
            raise RuntimeError("invalid checkpoint entropy iteration")
        self.last_iteration = last_iteration
