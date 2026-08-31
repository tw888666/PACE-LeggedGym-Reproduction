from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .constants import EFFORT_LIMIT, KD, KP, SATURATION_EFFORT, VELOCITY_LIMIT


@dataclass(frozen=True)
class ActuatorStep:
    q_encoder: torch.Tensor
    raw_pd_torque: torch.Tensor
    saturated_torque: torch.Tensor
    applied_torque: torch.Tensor


class TorqueDelayLine:
    """Exact integer-step FIFO: delay=d emits input[t-d], zero padded at reset."""

    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.delay_steps = int(delay_steps)
        self._buffer: Optional[torch.Tensor] = None
        self._head = 0

    def reset(self, reference: Optional[torch.Tensor] = None) -> None:
        self._head = 0
        if reference is None or self.delay_steps == 0:
            self._buffer = None
        else:
            self._buffer = torch.zeros(
                (self.delay_steps,) + tuple(reference.shape),
                dtype=reference.dtype,
                device=reference.device,
            )

    def compute(self, value: torch.Tensor) -> torch.Tensor:
        if self.delay_steps == 0:
            return value.clone()
        if self._buffer is None:
            self.reset(value)
        assert self._buffer is not None
        if self._buffer.shape[1:] != value.shape:
            raise ValueError(
                f"Delay-line shape changed from {self._buffer.shape[1:]} to {value.shape}; reset required"
            )
        delayed = self._buffer[self._head].clone()
        self._buffer[self._head].copy_(value)
        self._head = (self._head + 1) % self.delay_steps
        return delayed


class PACEActuatorCore:
    """Absolute-target PACE actuator core, independent of locomotion action processing."""

    def __init__(
        self,
        encoder_bias: torch.Tensor,
        delay_steps: int,
        kp: float = KP,
        kd: float = KD,
        saturation_effort: float = SATURATION_EFFORT,
        effort_limit: float = EFFORT_LIMIT,
        velocity_limit: float = VELOCITY_LIMIT,
    ) -> None:
        if encoder_bias.ndim != 1:
            raise ValueError("encoder_bias must be a one-dimensional joint vector")
        if saturation_effort <= 0 or effort_limit <= 0 or velocity_limit <= 0:
            raise ValueError("Motor limits must be positive")
        if effort_limit > saturation_effort:
            raise ValueError("effort_limit must not exceed saturation_effort")
        self.encoder_bias = encoder_bias.clone()
        self.delay_steps = int(delay_steps)
        self.kp = float(kp)
        self.kd = float(kd)
        self.saturation_effort = float(saturation_effort)
        self.effort_limit = float(effort_limit)
        self.velocity_limit = float(velocity_limit)
        self._delay = TorqueDelayLine(self.delay_steps)

    @property
    def velocity_corner(self) -> float:
        return self.velocity_limit * (1.0 + self.effort_limit / self.saturation_effort)

    def reset(self, reference: Optional[torch.Tensor] = None) -> None:
        self._delay.reset(reference)

    def _bias_for(self, value: torch.Tensor) -> torch.Tensor:
        bias = self.encoder_bias.to(device=value.device, dtype=value.dtype)
        if value.shape[-1] != bias.shape[0]:
            raise ValueError(
                f"Expected last dimension {bias.shape[0]}, got {value.shape[-1]}"
            )
        return bias

    def saturate(self, torque: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        if torque.shape != qdot.shape:
            raise ValueError(f"torque/qdot shape mismatch: {torque.shape} vs {qdot.shape}")
        clipped_velocity = torch.clamp(qdot, -self.velocity_corner, self.velocity_corner)
        max_torque = self.saturation_effort * (1.0 - clipped_velocity / self.velocity_limit)
        min_torque = self.saturation_effort * (-1.0 - clipped_velocity / self.velocity_limit)
        max_torque = torch.clamp(max_torque, -self.effort_limit, self.effort_limit)
        min_torque = torch.clamp(min_torque, -self.effort_limit, self.effort_limit)
        return torch.maximum(torch.minimum(torque, max_torque), min_torque)

    def step(
        self,
        absolute_q_target: torch.Tensor,
        q_true: torch.Tensor,
        qdot: torch.Tensor,
    ) -> ActuatorStep:
        if absolute_q_target.shape != q_true.shape or qdot.shape != q_true.shape:
            raise ValueError("absolute_q_target, q_true and qdot must have identical shapes")
        q_encoder = q_true - self._bias_for(q_true)
        raw_pd = self.kp * (absolute_q_target - q_encoder) - self.kd * qdot
        saturated = self.saturate(raw_pd, qdot)
        applied = self._delay.compute(saturated)
        return ActuatorStep(q_encoder, raw_pd, saturated, applied)

