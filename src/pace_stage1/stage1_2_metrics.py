"""Pure metric definitions for Stage 1.2 policy regularization."""

from __future__ import annotations

import math

import torch


ACTION_BOUND = 1.0
SCREENING_COEFFICIENTS = (0.0, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4)
TASK_VELOCITY_RMSE_THRESHOLD_M_S = 0.25


def coefficient_label(coefficient: float) -> str:
    labels = {
        0.0: "A0_lambda_0",
        1.0e-5: "A1_lambda_1e-5",
        3.0e-5: "A2_lambda_3e-5",
        1.0e-4: "A3_lambda_1e-4",
        3.0e-4: "A4_lambda_3e-4",
    }
    for candidate, label in labels.items():
        if math.isclose(coefficient, candidate, rel_tol=0.0, abs_tol=1.0e-12):
            return label
    raise ValueError(f"unsupported Stage1.2 coefficient: {coefficient}")


def raw_action_l2_penalty(
    raw_actions: torch.Tensor,
    coefficient: float,
    policy_dt_s: float,
) -> torch.Tensor:
    """Per-step penalty using the per-joint mean before action clipping."""
    if raw_actions.ndim != 2:
        raise ValueError("raw_actions must have shape [num_envs, num_actions]")
    if coefficient < 0.0:
        raise ValueError("coefficient must be non-negative")
    return coefficient * policy_dt_s * raw_actions.square().mean(dim=1)


def task_success_mask(
    survived_to_timeout: torch.Tensor,
    velocity_squared_error_sum: torch.Tensor,
    episode_length: torch.Tensor,
    threshold_m_s: float = TASK_VELOCITY_RMSE_THRESHOLD_M_S,
) -> torch.Tensor:
    """Joint survival-and-tracking success for fixed-vx evaluation."""
    if threshold_m_s <= 0.0:
        raise ValueError("threshold_m_s must be positive")
    rmse = torch.sqrt(velocity_squared_error_sum / episode_length.clamp_min(1))
    return survived_to_timeout.bool() & (rmse <= threshold_m_s)
