"""Single source of executable values for the Stage 1 task-only MDP."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

from pace_stage0.constants import CANONICAL_JOINT_NAMES


@dataclass(frozen=True)
class ObservationConfig:
    actor_dim: int = 48
    critic_dim: int = 353
    order: Tuple[str, ...] = (
        "base_linear_velocity_body[3]",
        "base_angular_velocity_body[3]",
        "projected_gravity_body[3]",
        "command_vx_vy_yaw_rate[3]",
        "encoder_joint_position[12]",
        "joint_velocity[12]",
        "previous_policy_action[12]",
    )
    component_slices: Tuple[Tuple[str, int, int], ...] = (
        ("base_linear_velocity_body", 0, 3),
        ("base_angular_velocity_body", 3, 6),
        ("projected_gravity_body", 6, 9),
        ("command_vx_vy_yaw_rate", 9, 12),
        ("encoder_joint_position", 12, 24),
        ("joint_velocity", 24, 36),
        ("previous_policy_action", 36, 48),
    )
    scales: Tuple[float, ...] = (
        *([2.0] * 3),
        *([0.25] * 3),
        *([1.0] * 3),
        2.0, 2.0, 0.25,
        *([1.0] * 12),
        *([0.05] * 12),
        *([1.0] * 12),
    )
    noise_half_ranges_after_scaling: Tuple[float, ...] = (
        *([0.2] * 3),
        *([0.05] * 3),
        *([0.05] * 3),
        *([0.0] * 3),
        *([0.01] * 12),
        *([0.075] * 12),
        *([0.0] * 12),
    )
    noise_distribution: str = "independent uniform [-half_range,+half_range]"
    normalization: str = "fixed component scales; no empirical running normalization"
    clip: float = 100.0
    critic_layout: Tuple[Tuple[str, int], ...] = (
        ("noise_free_actor_proprioception", 48),
        ("base_force_body", 3),
        ("base_torque_body", 3),
        ("ground_friction", 1),
        ("binary_foot_contacts", 4),
        ("base_centered_height_scan", 294),
    )
    height_scan_shape: Tuple[int, int] = (21, 14)
    height_scan_scale: float = 5.0
    height_scan_clip_m: Tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class ActionConfig:
    dim: int = 12
    joint_order: Tuple[str, ...] = CANONICAL_JOINT_NAMES
    clip: float = 100.0
    scale_rad: float = 0.5
    default_joint_pose_rad: Tuple[float, ...] = (
        0.0, 0.4, -0.8,
        0.0, 0.4, -0.8,
        0.0, -0.4, 0.8,
        0.0, -0.4, 0.8,
    )
    target_mapping: str = "q_target=frozen LocomotionTargetAdapter(action,q0,URDF limits)"
    target_semantics: str = "absolute joint-position target into frozen PACEActuatorCore"
    physics_dt_s: float = 0.0025
    policy_decimation: int = 4
    soft_limit_band_rad: float = math.radians(5.0)

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.policy_decimation


@dataclass(frozen=True)
class CommandConfig:
    dimensions: Tuple[str, ...] = (
        "lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_yaw_rad_s", "heading_rad"
    )
    lin_vel_x_range: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: Tuple[float, float] = (-1.0, 1.0)
    ang_vel_yaw_range: Tuple[float, float] = (-1.0, 1.0)
    heading_range: Tuple[float, float] = (-3.14, 3.14)
    heading_command: bool = True
    heading_gain: float = 0.5
    heading_yaw_rate_clip: Tuple[float, float] = (-1.0, 1.0)
    resampling_time_s: float = 10.0
    planar_deadband_m_s: float = 0.2
    command_curriculum: bool = False
    standing_semantics: str = "sampled planar norm <=0.2 becomes vx=vy=0; heading remains active"


@dataclass(frozen=True)
class RewardConfig:
    """Task-only terms and inherited LeggedGym aggregation semantics."""

    tracking_sigma: float = 0.25
    velocity_tracking_scale: float = 0.2
    collision_scale: float = -1.0
    foot_touchdown_scale: float = -0.1
    termination_scale: float = -0.0
    only_positive_rewards: bool = True
    foot_touchdown_half_life_iterations: float = 500.0
    foot_history_steps: int = 3
    foot_contact_threshold_n: float = 1.0
    collision_contact_threshold_n: float = 0.1
    termination_contact_threshold_n: float = 1.0
    reward_dt_s: float = 0.01
    aggregation_order: Tuple[str, ...] = (
        "scale non-termination terms by policy dt",
        "sum velocity_tracking, collision, foot_touchdown",
        "clip aggregate to >=0 when only_positive_rewards=true",
        "add non-timeout termination reward after clipping",
    )


@dataclass(frozen=True)
class ResetConfig:
    episode_length_s: float = 20.0
    timeout_comparison: str = "episode_length_steps > ceil(episode_length_s/policy_dt)"
    terminate_contact_bodies: Tuple[str, ...] = ("base",)
    collision_contact_bodies: Tuple[str, ...] = ("THIGH",)
    collision_also_joint_limit: bool = True
    initial_base_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.6)
    initial_base_quaternion_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    dof_position_multiplier_range: Tuple[float, float] = (0.5, 1.5)
    dof_reset_frame: str = "q_encoder=q0*uniform[0.5,1.5]; q_true=q_encoder+frozen encoder_bias"
    dof_velocity_rad_s: float = 0.0
    base_velocity_range: Tuple[float, float] = (-0.5, 0.5)
    rough_terrain_xy_offset_range_m: Tuple[float, float] = (-1.0, 1.0)
    reset_previous_action_to_zero: bool = True
    reset_actuator_delay_to_zero: bool = True
    resample_command_on_reset: bool = True
    reward_before_reset: bool = True
    timeout_bootstrap: bool = True


@dataclass(frozen=True)
class TerrainConfig:
    mesh_type: str = "trimesh"
    curriculum: bool = True
    horizontal_scale_m: float = 0.1
    vertical_scale_m: float = 0.005
    border_size_m: float = 25.0
    terrain_length_m: float = 8.0
    terrain_width_m: float = 8.0
    num_rows: int = 10
    num_cols: int = 20
    max_initial_level: int = 5
    terrain_proportions: Tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    terrain_types: Tuple[str, ...] = (
        "smooth_slope", "rough_slope", "stairs_up", "stairs_down", "discrete_boxes"
    )
    slope_threshold: float = 0.75
    nominal_static_friction: float = 1.0
    nominal_dynamic_friction: float = 1.0
    friction_range: Tuple[float, float] = (0.5, 1.25)
    restitution: float = 0.0


@dataclass(frozen=True)
class RandomizationConfig:
    randomize_ground_friction: bool = True
    randomize_terrain: bool = True
    randomize_dynamics: bool = False
    randomize_base_mass: bool = False
    randomize_motor_strength: bool = False
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_velocity_xy_m_s: float = 1.0
    initial_state_randomization: bool = True


@dataclass(frozen=True)
class PPOConfig:
    rsl_rl_version: str = "v1.0.2"
    seed: int = 1
    num_envs: int = 4096
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    num_steps_per_env: int = 24
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    max_iterations: int = 1500
    checkpoint_interval: int = 50


@dataclass(frozen=True)
class PPOSmokeConfig:
    classification: str = "SMOKE / NON-EXPERIMENTAL"
    num_envs: int = 2
    rollout_steps: int = 2
    optimizer_updates: int = 1
    seed: int = 123
    formal_seed: bool = False
    checkpoint_created: bool = False


@dataclass(frozen=True)
class FormalTaskOnlyBaselineConfig:
    experiment_name: str = "stage1_task_only_flat_seed0"
    num_envs: int = 4096
    max_iterations: int = 3000
    seed: int = 0
    terrain_mode: str = "plane"
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    friction_randomization: bool = True
    pushes: bool = True


@dataclass(frozen=True)
class Stage1Config:
    schema: str = "pace_stage1.task_only_mdp.v1"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ppo_smoke: PPOSmokeConfig = field(default_factory=PPOSmokeConfig)
    formal_baseline: FormalTaskOnlyBaselineConfig = field(
        default_factory=FormalTaskOnlyBaselineConfig
    )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


STAGE1_CONFIG = Stage1Config()


def ppo_train_cfg(config: Stage1Config = STAGE1_CONFIG) -> Dict[str, object]:
    """Return an unmodified rsl_rl v1.0.2 PPO parameter dictionary."""
    p = config.ppo
    return {
        "seed": p.seed,
        "runner_class_name": "OnPolicyRunner",
        "policy": {
            "init_noise_std": p.init_noise_std,
            "actor_hidden_dims": list(p.actor_hidden_dims),
            "critic_hidden_dims": list(p.critic_hidden_dims),
            "activation": p.activation,
        },
        "algorithm": {
            "value_loss_coef": p.value_loss_coef,
            "use_clipped_value_loss": p.use_clipped_value_loss,
            "clip_param": p.clip_param,
            "entropy_coef": p.entropy_coef,
            "num_learning_epochs": p.num_learning_epochs,
            "num_mini_batches": p.num_mini_batches,
            "learning_rate": p.learning_rate,
            "schedule": p.schedule,
            "gamma": p.gamma,
            "lam": p.lam,
            "desired_kl": p.desired_kl,
            "max_grad_norm": p.max_grad_norm,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": p.num_steps_per_env,
            "max_iterations": p.max_iterations,
            "save_interval": p.checkpoint_interval,
            "experiment_name": "pace_stage1_task_only_validation",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
    }
