"""Minimal bridge from PPO iteration count to the energy-off FTD schedule."""

from __future__ import annotations

from typing import Any

from .config import STAGE1_CONFIG, Stage1Config
from .semantics import foot_touchdown_schedule


def apply_task_iteration(
    environment: Any,
    iteration: int,
    config: Stage1Config = STAGE1_CONFIG,
) -> float:
    """Set the MDP clock once before a rollout; PPO parameters remain untouched."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    environment.set_training_iteration(iteration)
    return foot_touchdown_schedule(
        iteration, config.rewards.foot_touchdown_half_life_iterations
    )
