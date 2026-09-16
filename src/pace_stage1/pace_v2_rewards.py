"""Preregistered PACE-v2 task rewards, leaving the historical configuration intact."""

from dataclasses import replace

from .config import STAGE1_CONFIG, Stage1Config


REWARD_SEMANTICS_SPEC = "REWARD_SEMANTICS_RECONSTRUCTION_V1"


def pace_v2_environment_config(base: Stage1Config = STAGE1_CONFIG) -> Stage1Config:
    """Keep the inherited reward terms/time scale and preserve their signed sum."""
    return replace(
        base,
        rewards=replace(
            base.rewards,
            tracking_sigma=0.25,
            only_positive_rewards=False,
            reward_dt_s=base.action.policy_dt_s,
            aggregation_order=(
                "scale non-termination terms by policy dt",
                "sum velocity_tracking, collision, foot_touchdown",
                "preserve signed aggregate without nonnegative clipping",
                "add non-timeout termination reward after signed aggregation",
            ),
        ),
    )
