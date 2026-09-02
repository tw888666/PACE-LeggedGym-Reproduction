"""Isaac Gym Preview 4 VecEnv for the Stage 1 energy-off ablation MDP."""

from __future__ import annotations

# Isaac Gym must be imported before torch in Preview 4.
from isaacgym import gymapi, gymtorch

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from pace_stage0.constants import (
    ANYMAL_ASSET_FILE,
    ANYMAL_ASSET_ROOT,
    CANONICAL_JOINT_NAMES,
    EFFORT_LIMIT,
    VELOCITY_LIMIT,
    canonical_gather_indices,
)
from pace_stage0.decoder import decode_fit

from .config import STAGE1_CONFIG, Stage1Config
from .semantics import (
    BatchedPACEActuator,
    build_actor_observation,
    build_critic_observation,
    command_yaw_rate,
    command_resample_mask,
    compute_actuator_logging_metrics,
    compute_policy_target_pipeline,
    compute_task_reward_terms,
    make_target_adapter,
    quat_rotate_inverse,
    reset_true_joint_position,
    sample_commands,
    seed_everything,
    summarize_completed_episodes,
    timeout_mask,
)
from .terrain import TerrainMap, build_terrain


class Stage1LocomotionEnv:
    """rsl_rl v1.0.2-compatible vector environment.

    Production defaults are immutable in :mod:`pace_stage1.config`. Constructor
    overrides exist only so the pre-PPO CPU smoke can use a few plane environments.
    """

    def __init__(
        self,
        *,
        config: Stage1Config = STAGE1_CONFIG,
        num_envs: Optional[int] = None,
        sim_device: str = "cuda:0",
        headless: bool = True,
        terrain_mode: Optional[str] = None,
        seed: Optional[int] = None,
        actor_observation_noise: bool = True,
        enable_pushes: Optional[bool] = None,
    ) -> None:
        self.cfg = config
        self.num_envs = config.ppo.num_envs if num_envs is None else int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_obs = config.observation.actor_dim
        self.num_privileged_obs = config.observation.critic_dim
        self.num_actions = config.action.dim
        self.device = torch.device(sim_device)
        self.headless = bool(headless)
        self.terrain_mode = config.terrain.mesh_type if terrain_mode is None else terrain_mode
        if self.terrain_mode not in ("plane", "trimesh"):
            raise ValueError("Stage 1 terrain mode must be plane or trimesh")
        self.actor_observation_noise = bool(actor_observation_noise)
        self.enable_pushes = config.randomization.push_robots if enable_pushes is None else bool(enable_pushes)
        self.seed = config.ppo.seed if seed is None else int(seed)
        seed_everything(self.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.policy_dt = config.action.policy_dt_s
        if not math.isclose(self.policy_dt, config.rewards.reward_dt_s):
            raise ValueError("reward dt must equal the policy dt")
        self.max_episode_length_s = config.reset.episode_length_s
        self.max_episode_length = int(math.ceil(self.max_episode_length_s / self.policy_dt))
        self.command_resample_steps = int(config.commands.resampling_time_s / self.policy_dt)
        self.push_interval_steps = int(config.randomization.push_interval_s / self.policy_dt)
        self.common_step_counter = 0
        self.training_iteration = 0
        self._iteration_override: Optional[int] = None
        self.extras: Dict[str, object] = {}
        self._closed = False

        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        try:
            self._create_terrain()
            self._create_envs()
            self.gym.prepare_sim(self.sim)
            self._acquire_tensors()
            self._allocate_buffers()
            self.reset()
        except Exception:
            self.gym.destroy_sim(self.sim)
            self._closed = True
            raise

    def _create_sim(self):
        if self.device.type == "cuda":
            device_id = self.device.index or 0
            graphics_device = -1 if self.headless else device_id
        else:
            device_id = 0
            graphics_device = -1
        params = gymapi.SimParams()
        params.dt = self.cfg.action.physics_dt_s
        params.substeps = 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        params.use_gpu_pipeline = self.device.type == "cuda"
        params.physx.use_gpu = self.device.type == "cuda"
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 4
        params.physx.num_velocity_iterations = 0
        params.physx.num_threads = 10
        params.physx.contact_offset = 0.01
        params.physx.rest_offset = 0.0
        params.physx.bounce_threshold_velocity = 0.5
        params.physx.max_depenetration_velocity = 1.0
        params.physx.max_gpu_contact_pairs = 2**23
        params.physx.default_buffer_size_multiplier = 5.0
        sim = self.gym.create_sim(device_id, graphics_device, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError("gym.create_sim returned None")
        return sim

    def _create_terrain(self) -> None:
        self.terrain: Optional[TerrainMap] = None
        if self.terrain_mode == "plane":
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane.static_friction = self.cfg.terrain.nominal_static_friction
            plane.dynamic_friction = self.cfg.terrain.nominal_dynamic_friction
            plane.restitution = self.cfg.terrain.restitution
            self.gym.add_ground(self.sim, plane)
            return
        self.terrain = build_terrain(self.cfg.terrain)
        params = gymapi.TriangleMeshParams()
        params.nb_vertices = self.terrain.vertices.shape[0]
        params.nb_triangles = self.terrain.triangles.shape[0]
        params.transform.p.x = -self.cfg.terrain.border_size_m
        params.transform.p.y = -self.cfg.terrain.border_size_m
        params.transform.p.z = 0.0
        params.static_friction = self.cfg.terrain.nominal_static_friction
        params.dynamic_friction = self.cfg.terrain.nominal_dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            params,
        )

    def _asset_options(self):
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.disable_gravity = False
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        options.flip_visual_attachments = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.use_physx_armature = True
        options.linear_damping = 0.0
        options.angular_damping = 0.0
        options.max_linear_velocity = 1000.0
        options.max_angular_velocity = 1000.0
        options.thickness = 0.01
        return options

    def _make_origins(self) -> np.ndarray:
        if self.terrain is not None:
            levels = torch.randint(
                0,
                self.cfg.terrain.max_initial_level + 1,
                (self.num_envs,),
                generator=self.generator,
                device=self.device,
            )
            types = torch.div(
                torch.arange(self.num_envs, device=self.device) * self.cfg.terrain.num_cols,
                self.num_envs,
                rounding_mode="floor",
            ).to(torch.long)
            types = torch.clamp(types, max=self.cfg.terrain.num_cols - 1)
            self.terrain_levels = levels
            self.terrain_types = types
            return self.terrain.env_origins[levels.cpu().numpy(), types.cpu().numpy()].copy()
        cols = max(1, int(np.floor(np.sqrt(self.num_envs))))
        origins = np.zeros((self.num_envs, 3), dtype=np.float32)
        for index in range(self.num_envs):
            origins[index, 0] = 3.0 * (index // cols)
            origins[index, 1] = 3.0 * (index % cols)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return origins

    def _create_envs(self) -> None:
        asset = self.gym.load_asset(
            self.sim,
            str(ANYMAL_ASSET_ROOT),
            ANYMAL_ASSET_FILE,
            self._asset_options(),
        )
        if asset is None:
            raise RuntimeError("gym.load_asset returned None")
        self.asset = asset
        self.dof_names = tuple(self.gym.get_asset_dof_names(asset))
        self.gather_indices = canonical_gather_indices(self.dof_names)
        if tuple(self.dof_names[index] for index in self.gather_indices) != CANONICAL_JOINT_NAMES:
            raise RuntimeError("asset DOF order cannot be mapped to frozen canonical order")
        self.num_dof = self.gym.get_asset_dof_count(asset)
        self.body_names = tuple(self.gym.get_asset_rigid_body_names(asset))
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        if self.num_dof != self.num_actions:
            raise RuntimeError(f"expected 12 DOFs, got {self.num_dof}")

        fit = decode_fit()
        props = self.gym.get_asset_dof_properties(asset)
        gather = np.asarray(self.gather_indices, dtype=np.int64)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["friction"][gather] = fit.friction.astype(props["friction"].dtype)
        props["damping"][gather] = fit.damping.astype(props["damping"].dtype)
        props["armature"][gather] = fit.armature.astype(props["armature"].dtype)
        props["effort"].fill(EFFORT_LIMIT)
        props["velocity"].fill(VELOCITY_LIMIT)
        self.lower_limits = torch.as_tensor(props["lower"][gather], device=self.device, dtype=torch.float32)
        self.upper_limits = torch.as_tensor(props["upper"][gather], device=self.device, dtype=torch.float32)

        self.env_origins_np = self._make_origins()
        self.envs = []
        self.actors = []
        self.friction_coefficients = self.cfg.terrain.friction_range[0] + (
            self.cfg.terrain.friction_range[1] - self.cfg.terrain.friction_range[0]
        ) * torch.rand(self.num_envs, generator=self.generator, device=self.device)
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        per_row = max(1, int(np.sqrt(self.num_envs)))
        for index in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, per_row)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.env_origins_np[index])
            pose.p.z += self.cfg.reset.initial_base_position_m[2]
            actor = self.gym.create_actor(env, asset, pose, "anymal_d", index, 1, 0)
            self.gym.set_actor_dof_properties(env, actor, props)
            shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
            coefficient = float(self.friction_coefficients[index].item())
            for shape in shapes:
                shape.friction = coefficient
                shape.restitution = self.cfg.terrain.restitution
            self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
            self.envs.append(env)
            self.actors.append(actor)

        self.foot_indices = torch.tensor(
            [i for i, name in enumerate(self.body_names) if name.endswith("_SHANK")],
            dtype=torch.long,
            device=self.device,
        )
        self.thigh_indices = torch.tensor(
            [i for i, name in enumerate(self.body_names) if "THIGH" in name],
            dtype=torch.long,
            device=self.device,
        )
        self.base_indices = torch.tensor(
            [i for i, name in enumerate(self.body_names) if name == "base"],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.foot_indices) != 4 or len(self.base_indices) != 1:
            raise RuntimeError(
                f"unexpected collapsed-body mapping: names={self.body_names}, "
                f"feet={self.foot_indices.tolist()}, base={self.base_indices.tolist()}"
            )
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actors[0])
        self.body_mass = torch.tensor([p.mass for p in body_props], dtype=torch.float32, device=self.device)
        self.total_mass = float(self.body_mass.sum().item())
        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=self.device)
        self.actuator = BatchedPACEActuator(bias, fit.delay_steps)
        self.target_adapter = make_target_adapter(self.cfg)

    def _acquire_tensors(self) -> None:
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(self.num_envs, 13)
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, self.num_dof, 2)
        self.rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(
            self.num_envs, self.num_bodies, 13
        )
        self.contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(
            self.num_envs, self.num_bodies, 3
        )
        self.actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _allocate_buffers(self) -> None:
        self.env_origins = torch.as_tensor(self.env_origins_np, dtype=torch.float32, device=self.device)
        self.terrain_origins = None if self.terrain is None else torch.as_tensor(
            self.terrain.env_origins, dtype=torch.float32, device=self.device
        )
        self.default_dof_pos = torch.tensor(
            self.cfg.action.default_joint_pose_rad,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.applied_torque = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, 4), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.last_contacts = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.foot_speed_history = torch.zeros(
            (self.cfg.rewards.foot_history_steps, self.num_envs, 4), device=self.device
        )
        self.base_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_torque_world = torch.zeros((self.num_envs, 3), device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "velocity_tracking",
                "collision",
                "foot_touchdown",
                "termination",
                "total",
            )
        }
        self.episode_metric_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "linear_tracking_squared_error",
                "yaw_tracking_squared_error",
                "absolute_action",
                "torque_saturation_ratio",
                "mean_torque_utilization",
            )
        }
        x = torch.linspace(-1.5, 1.5, self.cfg.observation.height_scan_shape[0], device=self.device)
        y = torch.linspace(-1.0, 1.0, self.cfg.observation.height_scan_shape[1], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.height_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

    @property
    def dof_pos(self) -> torch.Tensor:
        gather = torch.as_tensor(self.gather_indices, dtype=torch.long, device=self.device)
        return self.dof_state[:, gather, 0]

    @property
    def dof_vel(self) -> torch.Tensor:
        gather = torch.as_tensor(self.gather_indices, dtype=torch.long, device=self.device)
        return self.dof_state[:, gather, 1]

    def _refresh(self) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = sample_commands(
            len(env_ids), self.generator, device=self.device, config=self.cfg
        )

    def _update_heading_commands(self) -> None:
        self.commands[:] = command_yaw_rate(self.commands, self.root_states[:, 3:7], self.cfg)

    def _push_robots(self) -> None:
        old_velocity = self.root_states[:, 7:9].clone()
        limit = self.cfg.randomization.max_push_velocity_xy_m_s
        new_velocity = -limit + 2.0 * limit * torch.rand(
            (self.num_envs, 2), generator=self.generator, device=self.device
        )
        self.root_states[:, 7:9] = new_velocity
        self.base_force_world[:, :2] = self.total_mass * (new_velocity - old_velocity) / self.policy_dt
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _height_scan(self) -> torch.Tensor:
        count = self.height_points.shape[0]
        if self.terrain is None:
            terrain_height = torch.zeros((self.num_envs, count), device=self.device)
        else:
            quaternion = self.root_states[:, 3:7]
            x, y, z, w = quaternion.unbind(dim=1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            px = self.height_points[:, 0].unsqueeze(0)
            py = self.height_points[:, 1].unsqueeze(0)
            world_x = self.root_states[:, 0:1] + cos_yaw[:, None] * px - sin_yaw[:, None] * py
            world_y = self.root_states[:, 1:2] + sin_yaw[:, None] * px + cos_yaw[:, None] * py
            ix = ((world_x + self.cfg.terrain.border_size_m) / self.terrain.horizontal_scale).long()
            iy = ((world_y + self.cfg.terrain.border_size_m) / self.terrain.horizontal_scale).long()
            heights = torch.as_tensor(self.terrain.height_field_raw, device=self.device)
            ix = torch.clamp(ix, 0, heights.shape[0] - 2)
            iy = torch.clamp(iy, 0, heights.shape[1] - 2)
            h1 = heights[ix, iy]
            h2 = heights[ix + 1, iy]
            h3 = heights[ix, iy + 1]
            terrain_height = torch.minimum(h1, torch.minimum(h2, h3)).float() * self.terrain.vertical_scale
        return self.root_states[:, 2:3] - 0.5 - terrain_height

    def _observations(self) -> None:
        quaternion = self.root_states[:, 3:7]
        base_lin = quat_rotate_inverse(quaternion, self.root_states[:, 7:10])
        base_ang = quat_rotate_inverse(quaternion, self.root_states[:, 10:13])
        gravity_world = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        projected_gravity = quat_rotate_inverse(quaternion, gravity_world)
        bias = self.actuator.encoder_bias.to(device=self.device, dtype=self.dof_pos.dtype)
        q_encoder = self.dof_pos - bias
        clean = build_actor_observation(
            base_lin,
            base_ang,
            projected_gravity,
            self.commands,
            q_encoder,
            self.dof_vel,
            self.actions,
            add_noise=False,
            config=self.cfg,
        )
        self.obs_buf = build_actor_observation(
            base_lin,
            base_ang,
            projected_gravity,
            self.commands,
            q_encoder,
            self.dof_vel,
            self.actions,
            add_noise=self.actor_observation_noise,
            generator=self.generator,
            config=self.cfg,
        )
        contacts = (
            self.contact_forces[:, self.foot_indices, 2]
            > self.cfg.rewards.foot_contact_threshold_n
        )
        contacts = contacts.clone()
        contacts[self.just_reset] = False
        force_body = quat_rotate_inverse(quaternion, self.base_force_world)
        torque_body = quat_rotate_inverse(quaternion, self.base_torque_world)
        self.privileged_obs_buf = build_critic_observation(
            clean,
            force_body,
            torque_body,
            self.friction_coefficients,
            contacts,
            self._height_scan(),
            self.cfg,
        )
        self.just_reset.zero_()

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        if (
            self.terrain is None
            or self.terrain_origins is None
            or not self.cfg.terrain.curriculum
            or self.common_step_counter == 0
            or len(env_ids) == 0
        ):
            return
        distance = torch.linalg.vector_norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.terrain.env_length / 2.0
        required = (
            torch.linalg.vector_norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        )
        move_down = (distance < required) & ~move_up
        self.terrain_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)
        solved = self.terrain_levels[env_ids] >= self.cfg.terrain.num_rows
        random_levels = torch.randint(
            0,
            self.cfg.terrain.num_rows,
            (len(env_ids),),
            generator=self.generator,
            device=self.device,
        )
        bounded = torch.clamp(self.terrain_levels[env_ids], min=0)
        self.terrain_levels[env_ids] = torch.where(solved, random_levels, bounded)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._update_terrain_curriculum(env_ids)
        completed = self.episode_length_buf[env_ids] > 0
        if torch.any(completed):
            completed_ids = env_ids[completed]
            self.extras["episode"] = {
                "rew_" + name: torch.mean(values[completed_ids]) / self.max_episode_length_s
                for name, values in self.episode_sums.items()
            }
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
            self.extras["episode"].update(
                summarize_completed_episodes(
                    self.episode_metric_sums["linear_tracking_squared_error"][completed_ids],
                    self.episode_metric_sums["yaw_tracking_squared_error"][completed_ids],
                    self.episode_metric_sums["absolute_action"][completed_ids],
                    self.episode_metric_sums["torque_saturation_ratio"][completed_ids],
                    self.episode_metric_sums["mean_torque_utilization"][completed_ids],
                    self.episode_length_buf[completed_ids],
                    self.time_out_buf[completed_ids],
                )
            )
        for values in self.episode_sums.values():
            values[env_ids] = 0.0
        for values in self.episode_metric_sums.values():
            values[env_ids] = 0.0
        count = len(env_ids)
        lo, hi = self.cfg.reset.dof_position_multiplier_range
        multiplier = lo + (hi - lo) * torch.rand(
            (count, self.num_dof), generator=self.generator, device=self.device
        )
        bias = self.actuator.encoder_bias.to(device=self.device, dtype=multiplier.dtype)
        canonical_q = reset_true_joint_position(
            self.default_dof_pos[env_ids], multiplier, bias
        )
        gather = torch.as_tensor(self.gather_indices, dtype=torch.long, device=self.device)
        self.dof_state[env_ids, :, 0] = 0.0
        self.dof_state[env_ids[:, None], gather[None, :], 0] = canonical_q
        self.dof_state[env_ids, :, 1] = self.cfg.reset.dof_velocity_rad_s

        self.root_states[env_ids] = 0.0
        self.root_states[env_ids, :3] = self.env_origins[env_ids]
        self.root_states[env_ids, 2] += self.cfg.reset.initial_base_position_m[2]
        if self.terrain is not None:
            xy_lo, xy_hi = self.cfg.reset.rough_terrain_xy_offset_range_m
            self.root_states[env_ids, :2] += xy_lo + (xy_hi - xy_lo) * torch.rand(
                (count, 2), generator=self.generator, device=self.device
            )
        self.root_states[env_ids, 3:7] = torch.tensor(
            self.cfg.reset.initial_base_quaternion_xyzw, device=self.device
        )
        vel_lo, vel_hi = self.cfg.reset.base_velocity_range
        self.root_states[env_ids, 7:13] = vel_lo + (vel_hi - vel_lo) * torch.rand(
            (count, 6), generator=self.generator, device=self.device
        )

        ids32 = env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ids32), count
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids32), count
        )
        self.actions[env_ids] = 0.0
        self.applied_torque[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.foot_speed_history[:, env_ids] = 0.0
        self.base_force_world[env_ids] = 0.0
        self.base_torque_world[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True
        self.just_reset[env_ids] = True
        self.time_out_buf[env_ids] = False
        self._resample_commands(env_ids)
        self.actuator.reset_envs(env_ids, self.actions)

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        self._refresh()
        self._update_heading_commands()
        self._observations()
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf

    def set_training_iteration(self, iteration: int) -> None:
        if iteration < 0:
            raise ValueError("training iteration must be non-negative")
        self.training_iteration = int(iteration)
        self._iteration_override = int(iteration)

    def step(self, actions: torch.Tensor):
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must have shape {(self.num_envs, self.num_actions)}")
        self.actions = torch.clamp(
            actions.to(self.device), -self.cfg.action.clip, self.cfg.action.clip
        )
        target_pipeline = compute_policy_target_pipeline(
            self.actions,
            self.default_dof_pos,
            self.lower_limits.repeat(self.num_envs, 1),
            self.upper_limits.repeat(self.num_envs, 1),
            self.cfg,
            self.target_adapter,
        )
        torque_saturation_ratio = torch.zeros(self.num_envs, device=self.device)
        mean_torque_utilization = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.cfg.action.policy_decimation):
            actuator_step = self.actuator.step(
                target_pipeline.selected_target, self.dof_pos, self.dof_vel
            )
            actuator_metrics = compute_actuator_logging_metrics(
                actuator_step.raw_pd_torque,
                actuator_step.saturated_torque,
                effort_limit_nm=self.actuator.effort_limit,
                saturation_epsilon_nm=(
                    self.cfg.training_diagnostics.torque_saturation_epsilon_nm
                ),
            )
            torque_saturation_ratio += actuator_metrics["torque_saturation_ratio"]
            mean_torque_utilization += actuator_metrics["mean_torque_utilization"]
            self.applied_torque = actuator_step.applied_torque
            asset_torque = torch.zeros_like(self.dof_state[:, :, 0])
            gather = torch.as_tensor(self.gather_indices, dtype=torch.long, device=self.device)
            asset_torque[:, gather] = self.applied_torque
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(asset_torque.reshape(-1))
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        torque_saturation_ratio /= self.cfg.action.policy_decimation
        mean_torque_utilization /= self.cfg.action.policy_decimation

        self._refresh()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self._iteration_override is None:
            self.training_iteration = (self.common_step_counter - 1) // self.cfg.ppo.num_steps_per_env
        self.base_force_world.zero_()
        self.base_torque_world.zero_()
        resample_ids = torch.nonzero(
            command_resample_mask(self.episode_length_buf, self.command_resample_steps),
            as_tuple=False,
        ).flatten()
        self._resample_commands(resample_ids)
        self._update_heading_commands()
        if self.enable_pushes and self.common_step_counter % self.push_interval_steps == 0:
            self._push_robots()

        base_contact = torch.any(
            torch.linalg.vector_norm(self.contact_forces[:, self.base_indices, :], dim=-1)
            > self.cfg.rewards.termination_contact_threshold_n,
            dim=1,
        )
        self.time_out_buf = timeout_mask(self.episode_length_buf, self.max_episode_length)
        self.reset_buf = base_contact | self.time_out_buf

        contacts = (
            self.contact_forces[:, self.foot_indices, 2]
            > self.cfg.rewards.foot_contact_threshold_n
        )
        foot_speed = torch.linalg.vector_norm(
            self.rigid_body_state[:, self.foot_indices, 7:10], dim=-1
        )
        self.foot_speed_history = torch.roll(self.foot_speed_history, shifts=1, dims=0)
        self.foot_speed_history[0] = foot_speed
        touchdown = contacts & ~self.last_contacts
        touchdown_speed = torch.max(self.foot_speed_history, dim=0).values * touchdown
        self.last_contacts = contacts
        thigh_contact = torch.any(
            torch.linalg.vector_norm(self.contact_forces[:, self.thigh_indices, :], dim=-1)
            > self.cfg.rewards.collision_contact_threshold_n,
            dim=1,
        ) if len(self.thigh_indices) else torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        joint_limit = torch.any(
            (self.dof_pos < self.lower_limits) | (self.dof_pos > self.upper_limits), dim=1
        ) if self.cfg.reset.collision_also_joint_limit else torch.zeros_like(thigh_contact)
        collision = thigh_contact | joint_limit
        quaternion = self.root_states[:, 3:7]
        base_lin = quat_rotate_inverse(quaternion, self.root_states[:, 7:10])
        base_ang = quat_rotate_inverse(quaternion, self.root_states[:, 10:13])
        terms = compute_task_reward_terms(
            self.commands,
            base_lin,
            base_ang,
            collision.to(torch.float32),
            touchdown_speed,
            self.reset_buf,
            self.time_out_buf,
            iteration=self.training_iteration,
            config=self.cfg,
        )
        self.rew_buf = terms.total
        self.episode_metric_sums["linear_tracking_squared_error"] += torch.sum(
            (self.commands[:, :2] - base_lin[:, :2]).square(), dim=1
        )
        self.episode_metric_sums["yaw_tracking_squared_error"] += (
            self.commands[:, 2] - base_ang[:, 2]
        ).square()
        self.episode_metric_sums["absolute_action"] += torch.mean(
            torch.abs(self.actions), dim=1
        )
        self.episode_metric_sums["torque_saturation_ratio"] += torque_saturation_ratio
        self.episode_metric_sums["mean_torque_utilization"] += mean_torque_utilization
        self.episode_sums["velocity_tracking"] += terms.scaled_velocity_tracking
        self.episode_sums["collision"] += terms.scaled_collision
        self.episode_sums["foot_touchdown"] += terms.scaled_foot_touchdown
        self.episode_sums["termination"] += terms.scaled_termination
        self.episode_sums["total"] += terms.total
        dones = self.reset_buf.clone()
        time_outs = self.time_out_buf.clone()
        self.extras = {
            "time_outs": time_outs,
            "reward_terms": {
                "velocity_tracking": terms.velocity_tracking,
                "collision": terms.collision,
                "foot_touchdown": terms.foot_touchdown,
                "termination": terms.termination,
                "task_before_clip": terms.task_before_clip,
                "task_after_clip": terms.task_after_clip,
            },
            "training_iteration": self.training_iteration,
        }
        self._reset_idx(torch.nonzero(dones, as_tuple=False).flatten())
        self._observations()
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, dones, self.extras

    def close(self) -> None:
        if not self._closed:
            self.gym.destroy_sim(self.sim)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
