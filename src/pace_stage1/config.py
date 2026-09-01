from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

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
        2.0, 2.0, 2.0,
        0.25, 0.25, 0.25,
        1.0, 1.0, 1.0,
        2.0, 2.0, 0.25,
        *([1.0] * 12),
        *([0.05] * 12),
        *([1.0] * 12),
    )
    noise_half_ranges_after_scaling: Tuple[float, ...] = (
        0.2, 0.2, 0.2,
        0.05, 0.05, 0.05,
        0.05, 0.05, 0.05,
        0.0, 0.0, 0.0,
        *([0.01] * 12),
        *([0.075] * 12),
        *([0.0] * 12),
    )
    noise_distribution: str = "independent uniform [-half_range,+half_range]"
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
    height_scan_extent_m: Tuple[float, float] = (3.0, 2.0)
    height_scan_resolution_nominal_m: float = 0.15
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
    target_mapping: str = "q_target = frozen LocomotionTargetAdapter(action,q0,URDF limits)"
    target_semantics: str = "absolute joint-position target into frozen PACEActuatorCore semantics"
    physics_dt_s: float = 0.0025
    policy_decimation: int = 4
    soft_limit_band_rad: float = math.radians(5.0)

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.policy_decimation


@dataclass(frozen=True)
class CommandConfig:
    dimensions: Tuple[str, ...] = ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_yaw_rad_s", "heading_rad")
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
    standing_semantics: str = "sampled planar norm <= 0.2 is set to vx=vy=0; yaw/heading is unchanged"


@dataclass(frozen=True)
class RewardConfig:
    tracking_sigma: float = 0.25
    velocity_tracking_scale: float = 0.2
    energy_scale: float = -16.0e-5
    collision_scale: float = -1.0
    foot_touchdown_scale: float = -0.1
    only_positive_rewards: bool = False
    foot_history_steps: int = 3
    contact_threshold_n: float = 1.0
    anymal_regeneration_coefficient: float = 0.0
    electrical_loss_coefficient_w_per_nm2: float = 1.0
    penalty_half_life_iterations: float = 500.0
    reward_integration: str = "instantaneous term * policy_dt; energy power thereby integrates to joules"


@dataclass(frozen=True)
class ResetConfig:
    episode_length_s: float = 20.0
    timeout_comparison: str = "episode_length_steps > ceil(episode_length_s/policy_dt)"
    terminate_contact_bodies: Tuple[str, ...] = ("base",)
    collision_contact_bodies: Tuple[str, ...] = ("THIGH",)
    collision_also_joint_limit: bool = True
    base_orientation_condition: str = "none; inherited termination is base contact"
    base_height_condition: str = "none; inherited termination is base contact"
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
    base_added_mass_range_kg: Tuple[float, float] = (-5.0, 5.0)
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
    entropy_initial: float = 0.01
    entropy_final: float = 0.001
    entropy_transition_iteration: float = 750.0
    entropy_tanh_rate: float = 0.01
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    num_steps_per_env: int = 24
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    max_iterations: int = 1500
    checkpoint_interval: int = 50
    runner_class_name: str = "OnPolicyRunner with Stage1 iteration schedule bridge"
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "PPO"
    seed_semantics: str = "seed Python, NumPy, Torch CPU and all CUDA devices before environment/model construction"


@dataclass(frozen=True)
class EvaluationConfig:
    protocol_name: str = "stage1_baseline_deterministic_v1"
    checkpoint_selection: str = "final iteration; no best-on-evaluation selection"
    policy_action: str = "deterministic actor mean"
    seeds: Tuple[int, ...] = (0, 1, 2)
    commands: Tuple[Tuple[float, float, float], ...] = (
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    )
    episodes_per_command_per_seed: int = 20
    actor_observation_noise: bool = False
    pushes: bool = False
    terrain_mode: str = "fixed plane first; frozen curriculum terrain suite reported separately"
    metrics: Tuple[str, ...] = (
        "return", "episode_length", "termination_rate", "timeout_rate",
        "linear_tracking_rmse", "yaw_tracking_rmse", "collision_rate",
        "foot_touchdown_speed", "energy_joule",
    )


@dataclass(frozen=True)
class Stage1Config:
    schema: str = "pace_stage1.config.v1"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


STAGE1_CONFIG = Stage1Config()


def ppo_train_cfg(config: Stage1Config = STAGE1_CONFIG) -> Dict[str, object]:
    """Return the rsl_rl v1.0.2-compatible baseline dictionary.

    PACE schedules are kept in a separate extension block because upstream PPO v1.0.2
    has no iteration-schedule fields. The Stage 1 bridge consumes that block.
    """
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
            "entropy_coef": p.entropy_initial,
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
            "policy_class_name": p.policy_class_name,
            "algorithm_class_name": p.algorithm_class_name,
            "num_steps_per_env": p.num_steps_per_env,
            "max_iterations": p.max_iterations,
            "save_interval": p.checkpoint_interval,
            "experiment_name": "pace_stage1_baseline",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
        "stage1_iteration_schedules": {
            "penalty_half_life_iterations": config.rewards.penalty_half_life_iterations,
            "entropy_initial": p.entropy_initial,
            "entropy_final": p.entropy_final,
            "entropy_transition_iteration": p.entropy_transition_iteration,
            "entropy_tanh_rate": p.entropy_tanh_rate,
        },
    }
