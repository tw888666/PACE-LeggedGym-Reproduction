"""Configuration identities for legacy, validation and formal PACE v2 runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .protocol_gate import (
    DEFAULT_PROTOCOL_MATRIX_PATH,
    formal_training_status,
    require_formal_training_allowed,
)


class ProtocolIdentity(str, Enum):
    LEG_GYM_REFERENCE = "LEG_GYM_REFERENCE"
    PACE_V2_VALIDATION = "PACE_V2_VALIDATION"
    PACE_V2_FORMAL = "PACE_V2_FORMAL"


@dataclass(frozen=True)
class PaceV2EntropyConfig:
    initial: float = 2.0e-3
    final: float = 5.0e-4
    turnover_iteration: int = 20_000
    slope_eta: Optional[float] = math.atanh(0.8) / 10_000


@dataclass(frozen=True)
class PaceV2PPOConfig:
    empirical_normalization: bool = True
    max_iterations: int = 30_000
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy: PaceV2EntropyConfig = field(default_factory=PaceV2EntropyConfig)
    num_learning_epochs: int = 5
    num_mini_batches: int = 10
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 1.0e-2
    max_grad_norm: float = 1.0
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (256, 256, 256, 128)
    init_noise_std: float = 1.5
    num_steps_per_env: int = 24
    activation: str = "elu"


@dataclass(frozen=True)
class ProtocolProfile:
    identity: ProtocolIdentity
    classification: str
    ppo: PaceV2PPOConfig
    requested_iterations: int
    formal: bool
    allows_short_validation: bool
    allows_formal_training: bool
    unresolved_parameters: Tuple[str, ...]


def leg_gym_reference_profile() -> ProtocolProfile:
    """Describe the historical branch without changing its executable config."""

    return ProtocolProfile(
        identity=ProtocolIdentity.LEG_GYM_REFERENCE,
        classification="HISTORICAL LEGGEDGYM-DERIVED ENERGY-OFF REFERENCE",
        ppo=PaceV2PPOConfig(
            empirical_normalization=False,
            max_iterations=3_000,
            entropy=PaceV2EntropyConfig(
                initial=1.0e-2,
                final=1.0e-2,
                turnover_iteration=0,
                slope_eta=None,
            ),
            num_mini_batches=4,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            init_noise_std=1.0,
        ),
        requested_iterations=3_000,
        formal=False,
        allows_short_validation=True,
        allows_formal_training=False,
        unresolved_parameters=(),
    )


def pace_v2_validation_profile(
    max_iterations: int = 500,
    matrix_path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> ProtocolProfile:
    """Return a non-formal profile; callers must explicitly resolve runtime assumptions."""

    if max_iterations <= 0 or max_iterations >= 30_000:
        raise ValueError("validation iterations must be in [1, 29999]")
    status = formal_training_status(matrix_path)
    return ProtocolProfile(
        identity=ProtocolIdentity.PACE_V2_VALIDATION,
        classification="PACE V2 INFRASTRUCTURE VALIDATION / NON-FORMAL",
        ppo=PaceV2PPOConfig(max_iterations=max_iterations),
        requested_iterations=max_iterations,
        formal=False,
        allows_short_validation=status.validation_allowed,
        allows_formal_training=False,
        unresolved_parameters=status.blocking_items,
    )


def pace_v2_formal_profile(
    matrix_path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> ProtocolProfile:
    """Instantiate the formal profile only when the matrix permits it."""

    require_formal_training_allowed(matrix_path)
    ppo = PaceV2PPOConfig()
    if ppo.entropy.slope_eta is None:
        raise RuntimeError(
            "PACE_V2_FORMAL cannot be instantiated while entropy slope eta is None"
        )
    return ProtocolProfile(
        identity=ProtocolIdentity.PACE_V2_FORMAL,
        classification="PACE V2 FORMAL",
        ppo=ppo,
        requested_iterations=30_000,
        formal=True,
        allows_short_validation=True,
        allows_formal_training=True,
        unresolved_parameters=(),
    )
