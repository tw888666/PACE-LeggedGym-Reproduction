"""Execution semantics for the Stage 1 PACE-derived energy-off ablation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from pace_stage0.actuator import PACEActuatorCore
from pace_stage0.target_adapter import LocomotionTargetAdapter

from .config import STAGE1_CONFIG, Stage1Config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def quat_rotate_inverse(quaternion_xyzw: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vec = quaternion_xyzw[..., :3]
    q_w = quaternion_xyzw[..., 3:4]
    return (
        vector * (2.0 * q_w.square() - 1.0)
        - torch.cross(q_vec, vector, dim=-1) * q_w * 2.0
        + q_vec * torch.sum(q_vec * vector, dim=-1, keepdim=True) * 2.0
    )


def yaw_from_quaternion(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def command_yaw_rate(
    commands: torch.Tensor,
    base_quaternion_xyzw: torch.Tensor,
    config: Stage1Config = STAGE1_CONFIG,
) -> torch.Tensor:
    result = commands.clone()
    if config.commands.heading_command:
        error = wrap_to_pi(commands[:, 3] - yaw_from_quaternion(base_quaternion_xyzw))
        lo, hi = config.commands.heading_yaw_rate_clip
        result[:, 2] = torch.clamp(config.commands.heading_gain * error, lo, hi)
    return result


def sample_commands(
    count: int,
    generator: torch.Generator,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    config: Stage1Config = STAGE1_CONFIG,
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    cfg = config.commands
    result = torch.empty((count, 4), dtype=dtype, device=device)

    def uniform(bounds: Tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(
            count, generator=generator, device=device, dtype=dtype
        )

    result[:, 0] = uniform(cfg.lin_vel_x_range)
    result[:, 1] = uniform(cfg.lin_vel_y_range)
    result[:, 2] = uniform(cfg.ang_vel_yaw_range)
    result[:, 3] = uniform(cfg.heading_range)
    moving = torch.linalg.vector_norm(result[:, :2], dim=1) > cfg.planar_deadband_m_s
    result[:, :2] *= moving.unsqueeze(1)
    return result


def build_actor_observation(
    base_linear_velocity_body: torch.Tensor,
    base_angular_velocity_body: torch.Tensor,
    projected_gravity_body: torch.Tensor,
    commands: torch.Tensor,
    encoder_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    add_noise: bool,
    generator: Optional[torch.Generator] = None,
    config: Stage1Config = STAGE1_CONFIG,
) -> torch.Tensor:
    pieces = (
        base_linear_velocity_body,
        base_angular_velocity_body,
        projected_gravity_body,
        commands[:, :3],
        encoder_joint_position,
        joint_velocity,
        previous_action,
    )
    if any(piece.ndim != 2 for piece in pieces):
        raise ValueError("all observation components must be rank-2 [env,component]")
    observation = torch.cat(pieces, dim=-1)
    if observation.shape[-1] != config.observation.actor_dim:
        raise ValueError(f"actor observation must have dimension {config.observation.actor_dim}")
    observation = observation * observation.new_tensor(config.observation.scales)
    if add_noise:
        half_range = observation.new_tensor(config.observation.noise_half_ranges_after_scaling)
        noise = 2.0 * torch.rand(
            observation.shape,
            generator=generator,
            dtype=observation.dtype,
            device=observation.device,
        ) - 1.0
        observation = observation + noise * half_range
    return torch.clamp(observation, -config.observation.clip, config.observation.clip)


def build_critic_observation(
    noise_free_actor_observation: torch.Tensor,
    base_force_body: torch.Tensor,
    base_torque_body: torch.Tensor,
    ground_friction: torch.Tensor,
    foot_contacts: torch.Tensor,
    height_scan_m: torch.Tensor,
    config: Stage1Config = STAGE1_CONFIG,
) -> torch.Tensor:
    scan_lo, scan_hi = config.observation.height_scan_clip_m
    scan = torch.clamp(height_scan_m, scan_lo, scan_hi) * config.observation.height_scan_scale
    result = torch.cat(
        (
            noise_free_actor_observation,
            base_force_body,
            base_torque_body,
            ground_friction.reshape(-1, 1),
            foot_contacts.to(dtype=noise_free_actor_observation.dtype),
            scan,
        ),
        dim=-1,
    )
    if result.shape[-1] != config.observation.critic_dim:
        raise ValueError(f"critic observation must have dimension {config.observation.critic_dim}")
    return torch.clamp(result, -config.observation.clip, config.observation.clip)


def foot_touchdown_schedule(iteration: float, half_life: float = 500.0) -> float:
    if iteration < 0 or half_life <= 0:
        raise ValueError("iteration must be non-negative and half_life positive")
    return 1.0 - math.exp(-math.log(2.0) * iteration / half_life)


def timeout_mask(episode_length_steps: torch.Tensor, maximum_steps: int) -> torch.Tensor:
    """LeggedGym uses a strict greater-than comparison."""
    return episode_length_steps > maximum_steps


def command_resample_mask(episode_length_steps: torch.Tensor, interval_steps: int) -> torch.Tensor:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    return episode_length_steps.remainder(interval_steps) == 0


def timeout_bootstrap(
    rewards: torch.Tensor,
    values: torch.Tensor,
    timeouts: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Exact rsl_rl v1.0.2 PPO.process_env_step timeout adjustment."""
    return rewards.clone() + gamma * values.reshape_as(rewards) * timeouts.to(rewards.dtype)


def reset_true_joint_position(
    default_command_position: torch.Tensor,
    multiplier: torch.Tensor,
    encoder_bias: torch.Tensor,
) -> torch.Tensor:
    if default_command_position.shape != multiplier.shape:
        raise ValueError("default position and multiplier must have identical shapes")
    return default_command_position * multiplier + encoder_bias


def summarize_completed_episodes(
    linear_tracking_squared_error_sum: torch.Tensor,
    yaw_tracking_squared_error_sum: torch.Tensor,
    absolute_action_sum: torch.Tensor,
    torque_saturation_ratio_sum: torch.Tensor,
    mean_torque_utilization_sum: torch.Tensor,
    episode_lengths: torch.Tensor,
    timeouts: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return logging-only completed-episode diagnostics without changing reward."""
    shapes = {
        tuple(linear_tracking_squared_error_sum.shape),
        tuple(yaw_tracking_squared_error_sum.shape),
        tuple(absolute_action_sum.shape),
        tuple(torque_saturation_ratio_sum.shape),
        tuple(mean_torque_utilization_sum.shape),
        tuple(episode_lengths.shape),
        tuple(timeouts.shape),
    }
    if len(shapes) != 1:
        raise ValueError("completed-episode diagnostic tensors must have identical shapes")
    if torch.any(episode_lengths <= 0):
        raise ValueError("completed episode lengths must be positive")
    lengths = episode_lengths.to(dtype=linear_tracking_squared_error_sum.dtype)
    return {
        "base_contact_rate": (~timeouts).to(lengths.dtype).mean(),
        "timeout_rate": timeouts.to(lengths.dtype).mean(),
        "velocity_tracking_rmse": torch.sqrt(
            linear_tracking_squared_error_sum / lengths
        ).mean(),
        "yaw_tracking_rmse": torch.sqrt(
            yaw_tracking_squared_error_sum / lengths
        ).mean(),
        "mean_abs_action": (absolute_action_sum / lengths).mean(),
        "torque_saturation_ratio": (
            torque_saturation_ratio_sum / lengths
        ).mean(),
        "mean_torque_utilization": (
            mean_torque_utilization_sum / lengths
        ).mean(),
    }


def compute_actuator_logging_metrics(
    commanded_torque: torch.Tensor,
    saturated_torque: torch.Tensor,
    *,
    effort_limit_nm: float,
    saturation_epsilon_nm: float,
) -> Dict[str, torch.Tensor]:
    """Return two aggregate actuator metrics without affecting control."""
    if commanded_torque.shape != saturated_torque.shape or commanded_torque.ndim != 2:
        raise ValueError("torques must have matching [env,dof] shapes")
    if effort_limit_nm <= 0.0 or saturation_epsilon_nm < 0.0:
        raise ValueError("effort limit must be positive and epsilon non-negative")
    saturated = (
        torch.abs(commanded_torque - saturated_torque)
        > saturation_epsilon_nm
    )
    return {
        "torque_saturation_ratio": saturated.to(commanded_torque.dtype).mean(dim=1),
        "mean_torque_utilization": (
            torch.abs(commanded_torque) / effort_limit_nm
        ).mean(dim=1),
    }


@dataclass(frozen=True)
class TaskRewardTerms:
    velocity_tracking: torch.Tensor
    collision: torch.Tensor
    foot_touchdown: torch.Tensor
    termination: torch.Tensor
    scaled_velocity_tracking: torch.Tensor
    scaled_collision: torch.Tensor
    scaled_foot_touchdown: torch.Tensor
    task_before_clip: torch.Tensor
    task_after_clip: torch.Tensor
    scaled_termination: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class PolicyTargetPipeline:
    before_joint_limit: torch.Tensor
    after_joint_limit: torch.Tensor
    selected_target: torch.Tensor
    saturation_mask: torch.Tensor


def compute_task_reward_terms(
    commands: torch.Tensor,
    base_linear_velocity_body: torch.Tensor,
    base_angular_velocity_body: torch.Tensor,
    collision_count: torch.Tensor,
    foot_touchdown_speed: torch.Tensor,
    terminated: torch.Tensor,
    timeouts: torch.Tensor,
    *,
    iteration: float,
    config: Stage1Config = STAGE1_CONFIG,
) -> TaskRewardTerms:
    """Aggregate reward in the same order as LeggedGym's compute_reward."""
    cfg = config.rewards
    linear_error = torch.sum(
        (commands[:, :2] - base_linear_velocity_body[:, :2]).square(), dim=1
    )
    yaw_error = (commands[:, 2] - base_angular_velocity_body[:, 2]).square()
    velocity = torch.exp(-linear_error / cfg.tracking_sigma) + torch.exp(
        -yaw_error / cfg.tracking_sigma
    )
    collision = collision_count.to(dtype=velocity.dtype)
    touchdown = torch.sum(foot_touchdown_speed, dim=1)
    termination = (terminated & ~timeouts).to(dtype=velocity.dtype)

    dt = cfg.reward_dt_s
    touchdown_factor = foot_touchdown_schedule(
        iteration, cfg.foot_touchdown_half_life_iterations
    )
    scaled_velocity = dt * cfg.velocity_tracking_scale * velocity
    scaled_collision = dt * cfg.collision_scale * collision
    scaled_touchdown = dt * touchdown_factor * cfg.foot_touchdown_scale * touchdown
    before_clip = scaled_velocity + scaled_collision + scaled_touchdown
    after_clip = torch.clamp(before_clip, min=0.0) if cfg.only_positive_rewards else before_clip
    scaled_termination = dt * cfg.termination_scale * termination
    total = after_clip + scaled_termination
    return TaskRewardTerms(
        velocity,
        collision,
        touchdown,
        termination,
        scaled_velocity,
        scaled_collision,
        scaled_touchdown,
        before_clip,
        after_clip,
        scaled_termination,
        total,
    )


class BatchedPACEActuator(PACEActuatorCore):
    """Only add per-environment FIFO reset; inherited actuator stepping is unchanged."""

    def reset_envs(self, env_ids: torch.Tensor, reference: torch.Tensor) -> None:
        if reference.ndim != 2:
            raise ValueError("reference must be [env,joint]")
        if self.delay_steps == 0:
            return
        if self._delay._buffer is None:  # pylint: disable=protected-access
            self.reset(reference)
        buffer = self._delay._buffer  # pylint: disable=protected-access
        assert buffer is not None
        if tuple(buffer.shape[1:]) != tuple(reference.shape):
            raise ValueError("actuator FIFO/reference shape mismatch")
        buffer[:, env_ids.to(device=buffer.device, dtype=torch.long), :] = 0.0


def make_target_adapter(config: Stage1Config = STAGE1_CONFIG) -> LocomotionTargetAdapter:
    return LocomotionTargetAdapter(
        action_scale=config.action.scale_rad,
        soft_band=config.action.soft_limit_band_rad,
    )


def compute_policy_target_pipeline(
    policy_action: torch.Tensor,
    default_dof_pos: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    config: Stage1Config = STAGE1_CONFIG,
    adapter: Optional[LocomotionTargetAdapter] = None,
) -> PolicyTargetPipeline:
    """Select the policy target without changing frozen actuator dynamics."""
    before = default_dof_pos + config.action.scale_rad * policy_action
    target_adapter = make_target_adapter(config) if adapter is None else adapter
    after = target_adapter(
        policy_action, default_dof_pos, lower_limits, upper_limits
    )
    saturation_mask = (
        torch.abs(before - after) > config.target_adapter.saturation_epsilon_rad
    )
    selected = (
        after
        if config.target_adapter.enforce_joint_limit_on_policy_target
        else before
    )
    return PolicyTargetPipeline(before, after, selected, saturation_mask)


def reward_manifest(config: Stage1Config = STAGE1_CONFIG) -> Tuple[Dict[str, object], ...]:
    return (
        {
            "name": "velocity_tracking",
            "definition": "exp(-planar_velocity_error/sigma)+exp(-yaw_rate_error/sigma)",
            "scale": config.rewards.velocity_tracking_scale,
        },
        {
            "name": "collision",
            "definition": "thigh contact or joint hard-limit violation indicator",
            "scale": config.rewards.collision_scale,
        },
        {
            "name": "foot_touchdown",
            "definition": "sum touchdown-foot max speed over current and previous two policy steps",
            "scale": config.rewards.foot_touchdown_scale,
        },
        {
            "name": "termination",
            "definition": "non-timeout termination indicator added after positive clipping",
            "scale": config.rewards.termination_scale,
        },
    )
