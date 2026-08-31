from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .actuator import PACEActuatorCore, TorqueDelayLine
from .constants import EFFORT_LIMIT, KD, KP, SATURATION_EFFORT, VELOCITY_LIMIT


PUBLIC_ENCODER_FEEDBACK = "public_encoder_feedback"
LEGACY_EFFECTIVE_BIAS = "legacy_effective_bias"
DIAGNOSTIC_BIAS_MODES = (PUBLIC_ENCODER_FEEDBACK, LEGACY_EFFECTIVE_BIAS)


@dataclass(frozen=True)
class DiagnosticActuatorStep:
    q_control: torch.Tensor
    raw_pd_torque: torch.Tensor
    saturated_torque: torch.Tensor
    applied_torque: torch.Tensor


class BiasLawDiagnosticActuator:
    """Diagnostic-only bias-law branch; never used by the formal actuator core."""

    def __init__(
        self,
        encoder_bias: torch.Tensor,
        delay_steps: int,
        mode: str,
        kp: float = KP,
        kd: float = KD,
        saturation_effort: float = SATURATION_EFFORT,
        effort_limit: float = EFFORT_LIMIT,
        velocity_limit: float = VELOCITY_LIMIT,
    ) -> None:
        if mode not in DIAGNOSTIC_BIAS_MODES:
            raise ValueError(f"Unknown diagnostic bias mode: {mode}")
        self.mode = mode
        self.encoder_bias = encoder_bias.clone()
        self._motor = PACEActuatorCore(
            encoder_bias=encoder_bias,
            delay_steps=delay_steps,
            kp=kp,
            kd=kd,
            saturation_effort=saturation_effort,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        )
        # Keep a separate delay line because PACEActuatorCore.step is deliberately
        # untouched. Public equivalence is enforced by unit tests.
        self._delay = TorqueDelayLine(delay_steps)

    @property
    def delay_steps(self) -> int:
        return self._motor.delay_steps

    def reset(self, reference: Optional[torch.Tensor] = None) -> None:
        self._delay.reset(reference)

    def _bias_for(self, value: torch.Tensor) -> torch.Tensor:
        bias = self.encoder_bias.to(device=value.device, dtype=value.dtype)
        if value.shape[-1] != bias.shape[0]:
            raise ValueError(
                f"Expected last dimension {bias.shape[0]}, got {value.shape[-1]}"
            )
        return bias

    def control_position(self, q_true: torch.Tensor) -> torch.Tensor:
        if self.mode == PUBLIC_ENCODER_FEEDBACK:
            return q_true - self._bias_for(q_true)
        return q_true.clone()

    def comparison_position(self, q_true: torch.Tensor) -> torch.Tensor:
        """Frozen Stage 0C frame, independent of the selected control law."""
        return q_true - self._bias_for(q_true)

    def step(
        self,
        absolute_q_target: torch.Tensor,
        q_true: torch.Tensor,
        qdot: torch.Tensor,
    ) -> DiagnosticActuatorStep:
        if absolute_q_target.shape != q_true.shape or qdot.shape != q_true.shape:
            raise ValueError("absolute_q_target, q_true and qdot must have identical shapes")
        q_control = self.control_position(q_true)
        raw_pd = (
            self._motor.kp * (absolute_q_target - q_control)
            - self._motor.kd * qdot
        )
        saturated = self._motor.saturate(raw_pd, qdot)
        applied = self._delay.compute(saturated)
        return DiagnosticActuatorStep(q_control, raw_pd, saturated, applied)
