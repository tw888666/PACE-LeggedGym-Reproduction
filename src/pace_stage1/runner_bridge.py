"""Small explicit bridge for PACE iteration schedules and rsl_rl v1.0.2."""

from __future__ import annotations

from typing import Any

from .config import STAGE1_CONFIG, Stage1Config
from .semantics import entropy_schedule


def apply_iteration_schedules(
    algorithm: Any,
    environment: Any,
    iteration: int,
    config: Stage1Config = STAGE1_CONFIG,
) -> float:
    """Set environment penalty time and upstream PPO's public entropy field.

    A formal training launcher must call this exactly once before each rollout.
    This function itself performs no rollout, update, checkpointing, or training.
    """
    environment.set_training_iteration(iteration)
    ppo = config.ppo
    coefficient = entropy_schedule(
        iteration,
        ppo.entropy_initial,
        ppo.entropy_final,
        ppo.entropy_transition_iteration,
        ppo.entropy_tanh_rate,
    )
    algorithm.entropy_coef = coefficient
    return coefficient
