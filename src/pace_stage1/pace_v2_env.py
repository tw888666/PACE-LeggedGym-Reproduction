"""Gated PACE v2 validation environment with preregistered control and task rewards."""

from __future__ import annotations

# Preserve Isaac Gym's required import order (before torch).
from .env import Stage1LocomotionEnv
from .pace_v2_control import PaceV2ControlMixin
from .pace_v2_profiles import pace_v2_validation_profile
from .config import STAGE1_CONFIG
from .pace_v2_rewards import pace_v2_environment_config
from .protocol_gate import require_validation_training_allowed


class PaceV2ValidationEnv(PaceV2ControlMixin, Stage1LocomotionEnv):
    """Separate control path; construction requires explicit validation approval."""

    def __init__(self, *, max_iterations: int = 500, config=STAGE1_CONFIG, **kwargs):
        require_validation_training_allowed()
        self.protocol_profile = pace_v2_validation_profile(max_iterations)
        super().__init__(config=pace_v2_environment_config(config), **kwargs)
