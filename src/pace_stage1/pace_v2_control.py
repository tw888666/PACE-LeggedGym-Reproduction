"""PACE v2 action-side control utilities that are disabled by default."""

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
    upper_active = (q_true >= upper_soft) & (q_target > upper)

    lower_fraction = torch.clamp((lower_soft - q_true) / band, min=0.0, max=1.0)
    lower_reshaped = q_target - lower_fraction * (q_target - lower)
    lower_active = (q_true <= lower_soft) & (q_target < lower)

    safe_target = torch.where(upper_active, upper_reshaped, q_target)
    return torch.where(lower_active, lower_reshaped, safe_target)
