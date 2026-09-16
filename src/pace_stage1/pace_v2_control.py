"""PACE v2 target control, isolated from the historical target adapter."""

from __future__ import annotations

from typing import Union

import torch


SoftBand = Union[float, torch.Tensor]


def pace_v2_hard_limit_safe_target(
    q_true: torch.Tensor,
    q_target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    soft_band: SoftBand,
) -> torch.Tensor:
    """Apply the target reshape from PACE v2 Eq. (9).

    This function does not clip the policy action or the target globally. It only
    reshapes an outward target when the measured joint position is inside the
    corresponding soft-limit band.
    """

    tensors = (q_true, q_target, lower, upper)
    if not (q_true.shape == q_target.shape == lower.shape == upper.shape):
        raise ValueError("q_true, q_target, lower and upper must have identical shapes")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("q_true, q_target, lower and upper must share one device")
    if not all(torch.is_floating_point(value) for value in tensors):
        raise TypeError("joint positions, targets and limits must be floating-point tensors")
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("joint positions, targets and limits must be finite")
    if torch.any(lower >= upper):
        raise ValueError("every lower limit must be smaller than its upper limit")

    band = torch.as_tensor(soft_band, dtype=q_target.dtype, device=q_target.device)
    try:
        band = torch.broadcast_to(band, q_target.shape)
    except RuntimeError as error:
        raise ValueError(
            "soft_band must be scalar or broadcastable to the joint tensors"
        ) from error
    if not torch.isfinite(band).all() or torch.any(band <= 0):
        raise ValueError("soft_band must be finite and strictly positive")
    if torch.any(2.0 * band > upper - lower):
        raise ValueError("soft-limit bands must not overlap")

    lower_soft = lower + band
    upper_soft = upper - band

    upper_fraction = torch.clamp(
        (q_true - upper_soft) / band, min=0.0, max=1.0
    )
    upper_reshaped = q_target - upper_fraction * (q_target - upper)
    # Enforce the exact endpoint despite rounding in (upper - upper_soft) / band.
    upper_reshaped = torch.where(q_true >= upper, upper, upper_reshaped)
    upper_active = (q_true >= upper_soft) & (q_target > upper)

    lower_fraction = torch.clamp((lower_soft - q_true) / band, min=0.0, max=1.0)
    lower_reshaped = q_target - lower_fraction * (q_target - lower)
    lower_reshaped = torch.where(q_true <= lower, lower, lower_reshaped)
    lower_active = (q_true <= lower_soft) & (q_target < lower)

    safe_target = torch.where(upper_active, upper_reshaped, q_target)
    return torch.where(lower_active, lower_reshaped, safe_target)


class PaceV2ControlMixin:
    """Control hooks for PACE v2; usable in component tests without Isaac Gym.

    The environment clips raw actions before constructing a held policy target.
    Eq. (9) reads the latest true joint state before every actuator/physics step.
    Encoder bias, torque saturation and FIFO delay remain inside the actuator.
    """

    def _compute_policy_target(self) -> torch.Tensor:
        return self.default_dof_pos + self.cfg.action.scale_rad * self.actions

    def _step_actuator(self, target: torch.Tensor):
        q_true = self.dof_pos
        safe_target = pace_v2_hard_limit_safe_target(
            q_true,
            target,
            self.lower_limits.expand_as(q_true),
            self.upper_limits.expand_as(q_true),
            self.cfg.action.soft_limit_band_rad,
        )
        return self.actuator.step(safe_target, q_true, self.dof_vel)
