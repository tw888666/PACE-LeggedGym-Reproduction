from __future__ import annotations

import torch

from .constants import HARD_LIMIT_SOFT_BAND


class LocomotionTargetAdapter:
    """Pure locomotion action-to-absolute-target adapter; never used by Stage 0 replay."""

    def __init__(self, action_scale: float = 0.5, soft_band: float = HARD_LIMIT_SOFT_BAND):
        if action_scale <= 0 or soft_band < 0:
            raise ValueError("action_scale must be positive and soft_band non-negative")
        self.action_scale = float(action_scale)
        self.soft_band = float(soft_band)

    def __call__(
        self,
        policy_action: torch.Tensor,
        default_dof_pos: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            policy_action.shape
            == default_dof_pos.shape
            == lower_limits.shape
            == upper_limits.shape
        ):
            raise ValueError("All adapter tensors must have identical shapes")
        safe_lower = lower_limits + self.soft_band
        safe_upper = upper_limits - self.soft_band
        if torch.any(safe_lower > safe_upper):
            raise ValueError("A joint range is narrower than twice the soft band")
        absolute_target = default_dof_pos + self.action_scale * policy_action
        return torch.maximum(torch.minimum(absolute_target, safe_upper), safe_lower)

