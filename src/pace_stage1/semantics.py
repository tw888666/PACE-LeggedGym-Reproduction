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
    """Apply the frozen seed to every RNG used by Stage 1."""
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
    a = vector * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, vector, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * vector, dim=-1, keepdim=True) * 2.0
    return a - b + c


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
        return lo + (hi - lo) * torch.rand(count, generator=generator, device=device, dtype=dtype)

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
    scales = observation.new_tensor(config.observation.scales)
    observation = observation * scales
    if add_noise:
        half_range = observation.new_tensor(config.observation.noise_half_ranges_after_scaling)
        noise = 2.0 * torch.rand(
            observation.shape,
            generator=generator,
            dtype=observation.dtype,
            device=observation.device,
        ) - 1.0
        observation = observation + noise * half_range
    return torch.clamp(
        observation,
        -config.observation.clip,
        config.observation.clip,
    )


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


def penalty_schedule(iteration: float, half_life: float = 500.0) -> float:
    if iteration < 0 or half_life <= 0:
        raise ValueError("iteration must be non-negative and half_life positive")
    return 1.0 - math.exp(-math.log(2.0) * iteration / half_life)


def entropy_schedule(
    iteration: float,
    initial: float,
    final: float,
    transition_iteration: float,
    tanh_rate: float,
) -> float:
    epsilon = 0.5 - 0.5 * math.tanh(tanh_rate * (iteration - transition_iteration))
    return final + epsilon * (initial - final)


def timeout_mask(episode_length_steps: torch.Tensor, maximum_steps: int) -> torch.Tensor:
    """Exact inherited LeggedGym timeout comparison (strictly greater-than)."""
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
    """rsl_rl v1.0.2 PPO.process_env_step timeout behavior."""
    return rewards.clone() + gamma * values.reshape_as(rewards) * timeouts.to(rewards.dtype)


def reset_true_joint_position(
    default_command_position: torch.Tensor,
    multiplier: torch.Tensor,
    encoder_bias: torch.Tensor,
) -> torch.Tensor:
    """Map inherited randomized encoder/command pose into Stage 0 true position."""
    if default_command_position.shape != multiplier.shape:
        raise ValueError("default position and multiplier must have identical shapes")
    return default_command_position * multiplier + encoder_bias


@dataclass(frozen=True)
class RewardTerms:
    velocity_tracking: torch.Tensor
    energy: torch.Tensor
    collision: torch.Tensor
    foot_touchdown: torch.Tensor
    total: torch.Tensor


def compute_reward_terms(
    commands: torch.Tensor,
    base_linear_velocity_body: torch.Tensor,
    base_angular_velocity_body: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    body_mass: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    collision_indicator: torch.Tensor,
    foot_touchdown_speed: torch.Tensor,
    *,
    iteration: float,
    config: Stage1Config = STAGE1_CONFIG,
) -> RewardTerms:
    cfg = config.rewards
    linear_error = torch.sum((commands[:, :2] - base_linear_velocity_body[:, :2]).square(), dim=1)
    yaw_error = (commands[:, 2] - base_angular_velocity_body[:, 2]).square()
    velocity = torch.exp(-linear_error / cfg.tracking_sigma) + torch.exp(-yaw_error / cfg.tracking_sigma)

    electrical = cfg.electrical_loss_coefficient_w_per_nm2 * torch.sum(applied_torque.square(), dim=1)
    signed_mechanical = torch.sum(applied_torque * joint_velocity, dim=1)
    mechanical = torch.where(
        signed_mechanical >= 0.0,
        signed_mechanical,
        cfg.anymal_regeneration_coefficient * signed_mechanical,
    )
    if body_linear_velocity_world.ndim != 3 or body_linear_velocity_world.shape[-1] != 3:
        raise ValueError("body_linear_velocity_world must be [env,body,3]")
    masses = body_mass.reshape(1, -1)
    if masses.shape[1] != body_linear_velocity_world.shape[1]:
        raise ValueError("body mass count does not match rigid-body velocities")
    # Paper defines v_b,z along -g, hence upward world-z velocity is negative here.
    potential = -9.81 * torch.sum(masses * body_linear_velocity_world[..., 2], dim=1)
    command_norm_sq = torch.sum(commands[:, :3].square(), dim=1)
    energy = (electrical + mechanical + potential) / (command_norm_sq + 1.0)

    collision = collision_indicator.to(dtype=velocity.dtype)
    touchdown = torch.sum(foot_touchdown_speed, dim=1)
    schedule = penalty_schedule(iteration, cfg.penalty_half_life_iterations)
    dt = config.action.policy_dt_s
    total = dt * (
        cfg.velocity_tracking_scale * velocity
        + cfg.collision_scale * collision
        + schedule * (cfg.energy_scale * energy + cfg.foot_touchdown_scale * touchdown)
    )
    return RewardTerms(velocity, energy, collision, touchdown, total)


class BatchedPACEActuator(PACEActuatorCore):
    """Stage 1-only per-environment reset extension of the frozen Stage 0 core.

    Torque computation and FIFO stepping are inherited byte-for-byte from
    :class:`PACEActuatorCore`. This class only zeroes selected batch columns of the
    already-created FIFO when individual locomotion environments reset.
    """

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


def reward_manifest(config: Stage1Config = STAGE1_CONFIG) -> Tuple[Dict[str, object], ...]:
    """Machine-facing summary used to cross-check provenance and implementation."""
    return (
        {
            "name": "velocity_tracking",
            "definition": "exp(-||vhat_xy-v_xy||^2/sigma)+exp(-(what_yaw-w_yaw)^2/sigma)",
            "scale": config.rewards.velocity_tracking_scale,
        },
        {
            "name": "energy",
            "definition": "(P_el+P_mech+P_pot)/(||command||^2+1)",
            "scale": config.rewards.energy_scale,
        },
        {
            "name": "collision",
            "definition": "indicator(any thigh contact or any joint outside URDF hard limit)",
            "scale": config.rewards.collision_scale,
        },
        {
            "name": "foot_touchdown",
            "definition": "sum over touchdown feet of max foot speed in current/previous two policy steps",
            "scale": config.rewards.foot_touchdown_scale,
        },
    )
