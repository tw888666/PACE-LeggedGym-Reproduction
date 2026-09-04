"""Stage 1.2 clip-conditioned raw-policy-action regularization experiments.

Every variant uses the same ``[-1, 1]`` actuator-interface action bound.  The
only scanned training variable is the coefficient on the mean squared sampled
action before clipping.  Energy and torque penalties remain disabled.
"""

from __future__ import annotations

# Isaac Gym Preview 4 must be imported before torch/rsl_rl.
from isaacgym import gymapi  # noqa: F401

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Dict, Optional

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from pace_stage0.constants import CANONICAL_JOINT_NAMES, PROJECT_ROOT
from pace_stage1.config import ppo_train_cfg
from pace_stage1.env import Stage1LocomotionEnv
from pace_stage1.semantics import quat_rotate_inverse
from pace_stage1.stage1_2_metrics import (
    ACTION_BOUND,
    TASK_VELOCITY_RMSE_THRESHOLD_M_S,
    coefficient_label,
    raw_action_l2_penalty,
    task_success_mask,
)


class Stage12PolicyEnv(Stage1LocomotionEnv):
    """Stage 1 environment with a hard action bound and raw-action preference."""

    def __init__(
        self,
        *,
        raw_action_l2_coefficient: float,
        fixed_command_vx: Optional[float] = None,
        collect_energy: bool = False,
        **kwargs,
    ):
        if raw_action_l2_coefficient < 0.0:
            raise ValueError("raw_action_l2_coefficient must be non-negative")
        self.raw_action_l2_coefficient = float(raw_action_l2_coefficient)
        self.fixed_command_vx = fixed_command_vx
        self.collect_energy = bool(collect_energy)
        super().__init__(**kwargs)

        self._actor_ref = None
        self._episode_regularization = torch.zeros(self.num_envs, device=self.device)
        self._iteration_scalar_sums: Dict[str, float] = {}
        self._iteration_clip_per_joint = torch.zeros(self.num_actions, device=self.device)
        self._iteration_steps = 0
        self._iteration_action_rows = 0
        self._iteration_actuator_rows = 0

        self._step_torque_saturation = torch.zeros(self.num_envs, device=self.device)
        self._step_torque_utilization = torch.zeros(self.num_envs, device=self.device)
        self._step_electrical_energy = torch.zeros(self.num_envs, device=self.device)
        self._step_mechanical_energy = torch.zeros(self.num_envs, device=self.device)
        self._step_potential_energy = torch.zeros(self.num_envs, device=self.device)
        self._step_actuator_substeps = 0
        original_step = self.actuator.step

        def observed_actuator_step(actuator_self, absolute_q_target, q_true, qdot):
            result = original_step(absolute_q_target, q_true, qdot)
            saturated = (
                (result.raw_pd_torque - result.saturated_torque).abs()
                > self.cfg.training_diagnostics.torque_saturation_epsilon_nm
            ).float()
            utilization = result.raw_pd_torque.abs() / self.actuator.effort_limit
            self._step_torque_saturation += saturated.mean(dim=1)
            self._step_torque_utilization += utilization.mean(dim=1)
            self._step_actuator_substeps += 1
            if self.collect_energy:
                dt = self.cfg.action.physics_dt_s
                self._step_electrical_energy += (
                    0.0192 * result.applied_torque.square().sum(dim=1) * dt
                )
                signed_mechanical = (result.applied_torque * qdot).sum(dim=1)
                self._step_mechanical_energy += torch.clamp(
                    signed_mechanical, min=0.0
                ) * dt
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                potential_power = (
                    self.body_mass.unsqueeze(0) * 9.81 * self.rigid_body_state[:, :, 9]
                ).sum(dim=1)
                self._step_potential_energy += potential_power * dt
            return result

        self.actuator.step = MethodType(observed_actuator_step, self.actuator)
        if self.fixed_command_vx is not None:
            self._set_fixed_command(torch.arange(self.num_envs, device=self.device))

    def bind_actor(self, actor) -> None:
        self._actor_ref = actor

    def _set_fixed_command(self, env_ids: torch.Tensor) -> None:
        if self.fixed_command_vx is None or len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 0] = self.fixed_command_vx

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if self.fixed_command_vx is None:
            super()._resample_commands(env_ids)
        else:
            self._set_fixed_command(env_ids)

    def _update_heading_commands(self) -> None:
        if self.fixed_command_vx is None:
            super()._update_heading_commands()
        else:
            self.commands[:, 0] = self.fixed_command_vx
            self.commands[:, 1:] = 0.0

    def _policy_snapshot(self, raw_actions: torch.Tensor):
        if self._actor_ref is None or self._actor_ref.distribution is None:
            return raw_actions.detach(), torch.zeros_like(raw_actions)
        return (
            self._actor_ref.action_mean.detach(),
            self._actor_ref.action_std.detach(),
        )

    def _add_scalar(self, name: str, value: float) -> None:
        self._iteration_scalar_sums[name] = (
            self._iteration_scalar_sums.get(name, 0.0) + float(value)
        )

    def step(self, actions: torch.Tensor):
        raw_actions = actions.to(self.device)
        policy_mean, policy_std = self._policy_snapshot(raw_actions)
        executed_actions = torch.clamp(raw_actions, -ACTION_BOUND, ACTION_BOUND)
        penalty = raw_action_l2_penalty(
            raw_actions, self.raw_action_l2_coefficient
        )
        targets = self.default_dof_pos + self.cfg.action.scale_rad * executed_actions

        quaternion = self.root_states[:, 3:7]
        base_lin = quat_rotate_inverse(quaternion, self.root_states[:, 7:10])
        base_ang = quat_rotate_inverse(quaternion, self.root_states[:, 10:13])
        velocity_squared_error = torch.sum(
            (self.commands[:, :2] - base_lin[:, :2]).square(), dim=1
        )
        yaw_squared_error = (self.commands[:, 2] - base_ang[:, 2]).square()

        self._step_torque_saturation.zero_()
        self._step_torque_utilization.zero_()
        self._step_electrical_energy.zero_()
        self._step_mechanical_energy.zero_()
        self._step_potential_energy.zero_()
        self._step_actuator_substeps = 0

        obs, critic_obs, rewards, dones, extras = super().step(executed_actions)
        if self._step_actuator_substeps != self.cfg.action.policy_decimation:
            raise RuntimeError("actuator observer missed a physics substep")
        self._step_torque_saturation /= self._step_actuator_substeps
        self._step_torque_utilization /= self._step_actuator_substeps
        self.rew_buf = rewards - penalty

        self._episode_regularization += penalty
        if torch.any(dones):
            extras.setdefault("episode", {})["raw_action_l2_penalty"] = (
                self._episode_regularization[dones].mean() / self.max_episode_length_s
            )
            self._episode_regularization[dones] = 0.0

        clip_mask = raw_actions.abs() > ACTION_BOUND
        mean_over_bound = policy_mean.abs() > ACTION_BOUND
        target_violation = (targets < self.lower_limits) | (targets > self.upper_limits)
        action_rows = raw_actions.shape[0]
        action_elements = raw_actions.numel()
        self._iteration_steps += 1
        self._iteration_action_rows += action_rows
        self._iteration_actuator_rows += action_rows * self._step_actuator_substeps
        self._iteration_clip_per_joint += clip_mask.float().sum(dim=0)
        self._add_scalar("policy_mean_abs", policy_mean.abs().sum().item())
        self._add_scalar("policy_mean_square", policy_mean.square().sum().item())
        self._add_scalar("policy_mean_over_bound", mean_over_bound.sum().item())
        self._add_scalar("policy_std", policy_std.sum().item())
        self._add_scalar("raw_sample_abs", raw_actions.abs().sum().item())
        self._add_scalar("raw_sample_square", raw_actions.square().sum().item())
        self._add_scalar("raw_sample_over_bound", clip_mask.sum().item())
        self._iteration_scalar_sums["raw_sample_max"] = max(
            self._iteration_scalar_sums.get("raw_sample_max", 0.0),
            raw_actions.abs().max().item(),
        )
        self._add_scalar("executed_abs", executed_actions.abs().sum().item())
        self._add_scalar("executed_square", executed_actions.square().sum().item())
        self._add_scalar("target_violation", target_violation.sum().item())
        self._iteration_scalar_sums["target_min"] = min(
            self._iteration_scalar_sums.get("target_min", float("inf")),
            targets.min().item(),
        )
        self._iteration_scalar_sums["target_max"] = max(
            self._iteration_scalar_sums.get("target_max", float("-inf")),
            targets.max().item(),
        )
        self._add_scalar("torque_saturation", self._step_torque_saturation.sum().item())
        self._add_scalar("torque_utilization", self._step_torque_utilization.sum().item())
        self._add_scalar("velocity_squared_error", velocity_squared_error.sum().item())
        self._add_scalar("yaw_squared_error", yaw_squared_error.sum().item())
        self._add_scalar("regularization_penalty", penalty.sum().item())
        extras["stage1_2"] = {
            "raw_actions": raw_actions,
            "executed_actions": executed_actions,
            "policy_mean": policy_mean,
            "policy_std": policy_std,
            "clip_mask": clip_mask,
            "targets": targets,
            "target_violation": target_violation,
            "torque_saturation_ratio": self._step_torque_saturation,
            "torque_utilization": self._step_torque_utilization,
            "regularization_penalty": penalty,
        }
        return obs, critic_obs, self.rew_buf, dones, extras

    def consume_iteration_metrics(self) -> Dict[str, object]:
        if self._iteration_steps == 0:
            return {}
        action_elements = self._iteration_action_rows * self.num_actions
        rows = self._iteration_action_rows
        values = self._iteration_scalar_sums
        metrics: Dict[str, object] = {
            "policy_mean_abs": values["policy_mean_abs"] / action_elements,
            "policy_mean_rms": math.sqrt(values["policy_mean_square"] / action_elements),
            "policy_mean_over_bound_ratio": values["policy_mean_over_bound"] / action_elements,
            "policy_std_mean": values["policy_std"] / action_elements,
            "raw_sample_action_abs_mean": values["raw_sample_abs"] / action_elements,
            "raw_sample_action_rms": math.sqrt(values["raw_sample_square"] / action_elements),
            "raw_sample_action_abs_max": values["raw_sample_max"],
            "executed_action_abs_mean": values["executed_abs"] / action_elements,
            "executed_action_rms": math.sqrt(values["executed_square"] / action_elements),
            "action_clip_ratio": values["raw_sample_over_bound"] / action_elements,
            "action_clip_ratio_per_joint": (
                self._iteration_clip_per_joint / rows
            ).detach().cpu().tolist(),
            "q_target_min_rad": values["target_min"],
            "q_target_max_rad": values["target_max"],
            "q_target_limit_violation_ratio": values["target_violation"] / action_elements,
            "torque_saturation_ratio": values["torque_saturation"] / rows,
            "mean_torque_utilization": values["torque_utilization"] / rows,
            "velocity_rmse_m_s": math.sqrt(values["velocity_squared_error"] / rows),
            "yaw_rate_rmse_rad_s": math.sqrt(values["yaw_squared_error"] / rows),
            "regularization_penalty_per_env_step": values["regularization_penalty"] / rows,
        }
        self._iteration_scalar_sums.clear()
        self._iteration_clip_per_joint.zero_()
        self._iteration_steps = 0
        self._iteration_action_rows = 0
        self._iteration_actuator_rows = 0
        return metrics


class Stage12Runner(OnPolicyRunner):
    """Adds per-iteration policy-distribution metrics without changing PPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.bind_actor(self.alg.actor_critic)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        metrics = self.env.consume_iteration_metrics()
        for name, value in metrics.items():
            if name == "action_clip_ratio_per_joint":
                for joint_name, joint_value in zip(CANONICAL_JOINT_NAMES, value):
                    self.writer.add_scalar(
                        f"Stage1.2/action_clip_ratio_joint/{joint_name}",
                        joint_value,
                        locs["it"],
                    )
            else:
                self.writer.add_scalar(f"Stage1.2/{name}", value, locs["it"])
        if metrics:
            print(
                "Stage1.2 iteration metrics: "
                f"mean={metrics['policy_mean_abs']:.4f} "
                f"sample={metrics['raw_sample_action_abs_mean']:.4f} "
                f"std={metrics['policy_std_mean']:.4f} "
                f"clip={metrics['action_clip_ratio']:.4f} "
                f"vx_rmse={metrics['velocity_rmse_m_s']:.4f} "
                f"tau_sat={metrics['torque_saturation_ratio']:.4f}",
                flush=True,
            )


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def train(
    coefficient: float,
    *,
    iterations: int,
    num_envs: int,
    seed: int,
    device: str,
    phase: str,
) -> Path:
    label = coefficient_label(coefficient)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "ppo"
        / "stage1_2_policy_regularization"
        / phase
        / label
        / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "gpt-训练清单.json"
    manifest: Dict[str, object] = {
        "schema": "pace_stage1.stage1_2_policy_regularization_train.v1",
        "classification": "STAGE1.2 PHYSICAL-POLICY CANDIDATE / NOT STAGE2 / NOT ECO",
        "phase": phase,
        "coefficient": coefficient,
        "penalty": "coefficient * mean_j(raw_sample_action_j^2)",
        "action_bound": [-ACTION_BOUND, ACTION_BOUND],
        "energy_reward": "OFF",
        "torque_l2": "OFF",
        "ppo_config_changed": False,
        "entropy_coefficient_changed": False,
        "training_commands": "unchanged Stage1 distribution",
        "terrain": "plane",
        "friction_randomization": True,
        "pushes": True,
        "seed": seed,
        "iterations": iterations,
        "num_envs": num_envs,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "state": "constructing_environment",
        "created_at_unix_s": time.time(),
    }
    _write_json(manifest_path, manifest)
    env = None
    try:
        env = Stage12PolicyEnv(
            raw_action_l2_coefficient=coefficient,
            num_envs=num_envs,
            sim_device=device,
            headless=True,
            terrain_mode="plane",
            seed=seed,
            actor_observation_noise=True,
            enable_pushes=True,
        )
        cfg = ppo_train_cfg()
        cfg["seed"] = seed
        cfg["runner"]["max_iterations"] = iterations
        cfg["runner"]["experiment_name"] = f"stage1_2_{phase}_{label}"
        runner = Stage12Runner(env, cfg, log_dir=str(run_dir), device=device)
        manifest["state"] = "training"
        _write_json(manifest_path, manifest)
        runner.learn(iterations, init_at_random_ep_len=True)
        checkpoint = run_dir / f"model_{iterations}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing final checkpoint: {checkpoint}")
        manifest.update(
            {
                "state": "completed",
                "checkpoint": str(checkpoint),
                "completed_at_unix_s": time.time(),
            }
        )
        _write_json(manifest_path, manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            {"state": "failed", "error": repr(error), "updated_at_unix_s": time.time()}
        )
        _write_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()


def _tensor_summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def evaluate(
    coefficient: float,
    checkpoint: Path,
    *,
    training_iterations: int,
    num_envs: int,
    steps: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    env = Stage12PolicyEnv(
        raw_action_l2_coefficient=coefficient,
        fixed_command_vx=1.0,
        collect_energy=True,
        num_envs=num_envs,
        sim_device=device,
        headless=True,
        terrain_mode="plane",
        seed=seed,
        actor_observation_noise=False,
        enable_pushes=False,
    )
    try:
        env.set_training_iteration(training_iterations)
        cfg = ppo_train_cfg()
        cfg["seed"] = seed
        runner = Stage12Runner(env, cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), load_optimizer=False)
        actor = runner.alg.actor_critic
        actor.eval()
        obs = env.get_observations()

        episode_length = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        episode_velocity_sq = torch.zeros(num_envs, device=env.device)
        episode_energy = torch.zeros(num_envs, device=env.device)
        completed_energy = []
        completed_velocity_rmse = []
        completed_task_success = []
        done_count = 0
        timeout_count = 0
        sums = {
            "velocity_squared_error": 0.0,
            "yaw_squared_error": 0.0,
            "policy_mean_abs": 0.0,
            "policy_mean_square": 0.0,
            "policy_mean_over_bound": 0.0,
            "policy_std": 0.0,
            "sample_abs": 0.0,
            "sample_square": 0.0,
            "sample_max": 0.0,
            "sample_over_bound": 0.0,
            "executed_abs": 0.0,
            "executed_square": 0.0,
            "target_min": float("inf"),
            "target_max": float("-inf"),
            "target_violation": 0.0,
            "torque_saturation": 0.0,
            "torque_utilization": 0.0,
            "electrical_j": 0.0,
            "mechanical_j": 0.0,
            "potential_j": 0.0,
        }
        clip_per_joint = torch.zeros(env.num_actions, device=env.device)
        started = time.time()
        with torch.inference_mode():
            for step_index in range(steps):
                quaternion = env.root_states[:, 3:7]
                base_lin = quat_rotate_inverse(quaternion, env.root_states[:, 7:10])
                base_ang = quat_rotate_inverse(quaternion, env.root_states[:, 10:13])
                velocity_sq = (base_lin[:, 0] - 1.0).square() + base_lin[:, 1].square()
                yaw_sq = base_ang[:, 2].square()

                actor.update_distribution(obs)
                policy_mean = actor.action_mean
                policy_std = actor.action_std
                sampled_actions = actor.distribution.sample()
                deterministic_raw = policy_mean
                executed = torch.clamp(deterministic_raw, -ACTION_BOUND, ACTION_BOUND)
                targets = env.default_dof_pos + env.cfg.action.scale_rad * executed
                target_violation = (targets < env.lower_limits) | (targets > env.upper_limits)
                env.bind_actor(actor)
                obs, _, _, dones, extras = env.step(deterministic_raw)

                episode_length += 1
                episode_velocity_sq += velocity_sq
                step_energy = (
                    env._step_electrical_energy
                    + env._step_mechanical_energy
                    + env._step_potential_energy
                )
                episode_energy += step_energy
                sums["velocity_squared_error"] += velocity_sq.sum().item()
                sums["yaw_squared_error"] += yaw_sq.sum().item()
                sums["policy_mean_abs"] += policy_mean.abs().sum().item()
                sums["policy_mean_square"] += policy_mean.square().sum().item()
                sums["policy_mean_over_bound"] += (
                    policy_mean.abs() > ACTION_BOUND
                ).sum().item()
                sums["policy_std"] += policy_std.sum().item()
                sums["sample_abs"] += sampled_actions.abs().sum().item()
                sums["sample_square"] += sampled_actions.square().sum().item()
                sums["sample_max"] = max(
                    sums["sample_max"], sampled_actions.abs().max().item()
                )
                sample_clip = sampled_actions.abs() > ACTION_BOUND
                sums["sample_over_bound"] += sample_clip.sum().item()
                clip_per_joint += sample_clip.float().sum(dim=0)
                sums["executed_abs"] += executed.abs().sum().item()
                sums["executed_square"] += executed.square().sum().item()
                sums["target_min"] = min(sums["target_min"], targets.min().item())
                sums["target_max"] = max(sums["target_max"], targets.max().item())
                sums["target_violation"] += target_violation.sum().item()
                sums["torque_saturation"] += extras["stage1_2"][
                    "torque_saturation_ratio"
                ].sum().item()
                sums["torque_utilization"] += extras["stage1_2"][
                    "torque_utilization"
                ].sum().item()
                sums["electrical_j"] += env._step_electrical_energy.sum().item()
                sums["mechanical_j"] += env._step_mechanical_energy.sum().item()
                sums["potential_j"] += env._step_potential_energy.sum().item()

                if torch.any(dones):
                    survived = extras["time_outs"][dones]
                    lengths = episode_length[dones]
                    velocity_error = episode_velocity_sq[dones]
                    completed_velocity_rmse.append(
                        torch.sqrt(velocity_error / lengths.clamp_min(1)).cpu()
                    )
                    completed_task_success.append(
                        task_success_mask(survived, velocity_error, lengths).cpu()
                    )
                    completed_energy.append(episode_energy[dones].cpu())
                    done_count += int(dones.sum().item())
                    timeout_count += int(survived.sum().item())
                    episode_length[dones] = 0
                    episode_velocity_sq[dones] = 0.0
                    episode_energy[dones] = 0.0
                if (step_index + 1) % 500 == 0 or step_index + 1 == steps:
                    print(
                        f"eval coefficient={coefficient:g} step={step_index + 1}/{steps}",
                        flush=True,
                    )

        sample_count = num_envs * steps
        action_count = sample_count * env.num_actions
        duration_s = steps * env.policy_dt
        energy = torch.cat(completed_energy) if completed_energy else torch.empty(0)
        episode_rmse = (
            torch.cat(completed_velocity_rmse)
            if completed_velocity_rmse
            else torch.empty(0)
        )
        successes = (
            torch.cat(completed_task_success)
            if completed_task_success
            else torch.empty(0, dtype=torch.bool)
        )
        return {
            "schema": "pace_stage1.stage1_2_policy_regularization_eval.v1",
            "classification": "STAGE1.2 PHYSICAL-POLICY CANDIDATE / NOT STAGE2 / NOT ECO",
            "coefficient": coefficient,
            "checkpoint": str(checkpoint),
            "training_iterations": training_iterations,
            "fixed_command_vx_m_s": 1.0,
            "terrain": "plane",
            "seed": seed,
            "num_envs": num_envs,
            "policy_steps": steps,
            "duration_per_env_s": duration_s,
            "completed_episodes": done_count,
            "timeout_survival_rate": timeout_count / done_count if done_count else None,
            "task_success_definition": (
                "survived_to_timeout AND episode_velocity_rmse_m_s <= 0.25"
            ),
            "task_success_rate": successes.float().mean().item() if successes.numel() else None,
            "velocity_rmse_m_s": math.sqrt(sums["velocity_squared_error"] / sample_count),
            "yaw_rate_rmse_rad_s": math.sqrt(sums["yaw_squared_error"] / sample_count),
            "completed_episode_velocity_rmse_m_s": (
                _tensor_summary(episode_rmse) if episode_rmse.numel() else None
            ),
            "policy_mean_abs": sums["policy_mean_abs"] / action_count,
            "policy_mean_rms": math.sqrt(sums["policy_mean_square"] / action_count),
            "policy_mean_over_bound_ratio": sums["policy_mean_over_bound"] / action_count,
            "policy_std_mean": sums["policy_std"] / action_count,
            "sampled_raw_action_abs_mean": sums["sample_abs"] / action_count,
            "sampled_raw_action_rms": math.sqrt(sums["sample_square"] / action_count),
            "sampled_raw_action_abs_max": sums["sample_max"],
            "sampled_action_clip_ratio": sums["sample_over_bound"] / action_count,
            "sampled_action_clip_ratio_per_joint": dict(
                zip(
                    CANONICAL_JOINT_NAMES,
                    (clip_per_joint / sample_count).cpu().tolist(),
                )
            ),
            "executed_action_abs_mean": sums["executed_abs"] / action_count,
            "executed_action_rms": math.sqrt(sums["executed_square"] / action_count),
            "q_target_range_rad": [sums["target_min"], sums["target_max"]],
            "q_target_limit_violation_ratio": sums["target_violation"] / action_count,
            "torque_saturation_ratio": sums["torque_saturation"] / sample_count,
            "mean_torque_utilization": sums["torque_utilization"] / sample_count,
            "completed_episode_total_energy_j": (
                _tensor_summary(energy) if energy.numel() else None
            ),
            "power_population_time_average_w": {
                "electrical": sums["electrical_j"] / (num_envs * duration_s),
                "mechanical_no_regen": sums["mechanical_j"] / (num_envs * duration_s),
                "potential": sums["potential_j"] / (num_envs * duration_s),
                "total": (
                    sums["electrical_j"]
                    + sums["mechanical_j"]
                    + sums["potential_j"]
                )
                / (num_envs * duration_s),
            },
            "wall_time_s": time.time() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=("screening", "formal"), default="screening")
    parser.add_argument("--eval-num-envs", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=2001)
    args = parser.parse_args()
    coefficient_label(args.coefficient)
    run_dir = train(
        args.coefficient,
        iterations=args.iterations,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        phase=args.phase,
    )
    checkpoint = run_dir / f"model_{args.iterations}.pt"
    result = evaluate(
        args.coefficient,
        checkpoint,
        training_iterations=args.iterations,
        num_envs=args.eval_num_envs,
        steps=args.eval_steps,
        seed=args.seed,
        device=args.device,
    )
    output_path = run_dir / "gpt-固定速度确定性评估.json"
    _write_json(output_path, result)
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"EVAL={output_path}", flush=True)


if __name__ == "__main__":
    main()
