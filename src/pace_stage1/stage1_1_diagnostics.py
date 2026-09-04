"""Independent Stage 1.1 action/torque regularization diagnostics.

This module deliberately leaves the Stage 1 baseline configuration and reward
implementation unchanged.  It provides three short, fixed-command diagnostic
variants around :class:`Stage1LocomotionEnv`.
"""

from __future__ import annotations

# Isaac Gym Preview 4 must be imported before torch/rsl_rl.
from isaacgym import gymapi  # noqa: F401

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Dict, Optional

import numpy as np
import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from pace_stage0.constants import PROJECT_ROOT
from pace_stage1.config import STAGE1_CONFIG, ppo_train_cfg
from pace_stage1.env import Stage1LocomotionEnv
from pace_stage1.semantics import quat_rotate_inverse


MODES = ("action_clip", "action_l2", "torque_l2")
MODE_LABELS = {
    "action_clip": "action clipping only / 动作裁剪",
    "action_l2": "action L2 regularization / 动作二范数正则",
    "torque_l2": "torque L2 regularization / 力矩二范数正则",
}


@dataclass(frozen=True)
class DiagnosticSpec:
    mode: str
    action_bound: Optional[float] = None
    action_l2_coefficient: float = 0.0
    torque_l2_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown diagnostic mode: {self.mode}")


def diagnostic_spec(mode: str) -> DiagnosticSpec:
    if mode == "action_clip":
        return DiagnosticSpec(mode=mode, action_bound=1.0)
    if mode == "action_l2":
        return DiagnosticSpec(mode=mode, action_l2_coefficient=1.0e-4)
    if mode == "torque_l2":
        return DiagnosticSpec(mode=mode, torque_l2_coefficient=1.0e-5)
    raise ValueError(f"unknown diagnostic mode: {mode}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


class Stage11DiagnosticEnv(Stage1LocomotionEnv):
    """Fixed-vx Stage 1 environment with one isolated diagnostic intervention."""

    def __init__(self, *, diagnostic: DiagnosticSpec, collect_energy: bool = False, **kwargs):
        self.diagnostic = diagnostic
        self.collect_energy = bool(collect_energy)
        super().__init__(**kwargs)
        self._torque_square_sum = torch.zeros(self.num_envs, device=self.device)
        self._electrical_energy = torch.zeros_like(self._torque_square_sum)
        self._mechanical_energy = torch.zeros_like(self._torque_square_sum)
        self._potential_energy = torch.zeros_like(self._torque_square_sum)
        self._saturation_sum = torch.zeros_like(self._torque_square_sum)
        self._actuator_substeps = 0
        original_step = self.actuator.step

        def observed_step(actuator_self, absolute_q_target, q_true, qdot):
            result = original_step(absolute_q_target, q_true, qdot)
            self._torque_square_sum += result.applied_torque.square().sum(dim=1)
            self._saturation_sum += (
                (result.raw_pd_torque - result.saturated_torque).abs()
                > self.cfg.training_diagnostics.torque_saturation_epsilon_nm
            ).float().mean(dim=1)
            self._actuator_substeps += 1
            if self.collect_energy:
                dt = self.cfg.action.physics_dt_s
                self._electrical_energy += 0.0192 * result.applied_torque.square().sum(dim=1) * dt
                signed_mechanical = (result.applied_torque * qdot).sum(dim=1)
                self._mechanical_energy += torch.clamp(signed_mechanical, min=0.0) * dt
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                potential_power = (
                    self.body_mass.unsqueeze(0) * 9.81 * self.rigid_body_state[:, :, 9]
                ).sum(dim=1)
                self._potential_energy += potential_power * dt
            return result

        self.actuator.step = MethodType(observed_step, self.actuator)

    def _set_fixed_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 0] = 1.0

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        self._set_fixed_command(env_ids)

    def _update_heading_commands(self) -> None:
        self.commands[:, 0] = 1.0
        self.commands[:, 1:] = 0.0

    def step(self, actions: torch.Tensor):
        raw_actions = actions.to(self.device)
        effective_actions = raw_actions
        if self.diagnostic.action_bound is not None:
            bound = self.diagnostic.action_bound
            effective_actions = torch.clamp(raw_actions, -bound, bound)

        action_penalty = (
            self.cfg.action.policy_dt_s
            * self.diagnostic.action_l2_coefficient
            * effective_actions.square().sum(dim=1)
        )
        self._torque_square_sum.zero_()
        self._electrical_energy.zero_()
        self._mechanical_energy.zero_()
        self._potential_energy.zero_()
        self._saturation_sum.zero_()
        self._actuator_substeps = 0
        obs, critic_obs, rewards, dones, extras = super().step(effective_actions)
        if self._actuator_substeps != self.cfg.action.policy_decimation:
            raise RuntimeError("actuator observer missed a physics substep")
        mean_substep_torque_square = self._torque_square_sum / self._actuator_substeps
        torque_penalty = (
            self.cfg.action.policy_dt_s
            * self.diagnostic.torque_l2_coefficient
            * mean_substep_torque_square
        )
        regularization_penalty = action_penalty + torque_penalty
        self.rew_buf = rewards - regularization_penalty
        extras["diagnostic"] = {
            "raw_action_mean_abs": raw_actions.abs().mean(dim=1),
            "effective_action_mean_abs": effective_actions.abs().mean(dim=1),
            "regularization_penalty": regularization_penalty,
            "torque_saturation_ratio": self._saturation_sum / self._actuator_substeps,
        }
        return obs, critic_obs, self.rew_buf, dones, extras


def train(mode: str, iterations: int, num_envs: int, seed: int, device: str) -> Path:
    spec = diagnostic_spec(mode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "artifacts" / "ppo" / "stage1_1_diagnostics" / mode / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "gpt-训练清单.json"
    manifest = {
        "schema": "pace_stage1.stage1_1_diagnostic_train.v1",
        "classification": "DIAGNOSTIC ONLY / NOT A BASELINE / NOT ECO",
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "diagnostic": asdict(spec),
        "fixed_command_vx_m_s": 1.0,
        "fixed_command_vy_m_s": 0.0,
        "fixed_command_yaw_rate_rad_s": 0.0,
        "terrain": "plane",
        "seed": seed,
        "iterations": iterations,
        "num_envs": num_envs,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "baseline_reward_source_unchanged": "src/pace_stage1/semantics.py",
        "baseline_environment_source": "src/pace_stage1/env.py",
        "diagnostic_source": "src/pace_stage1/stage1_1_diagnostics.py",
        "diagnostic_source_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_created": False,
        "state": "starting",
        "created_at_unix_s": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    env = None
    try:
        env = Stage11DiagnosticEnv(
            diagnostic=spec,
            collect_energy=False,
            num_envs=num_envs,
            sim_device=device,
            headless=True,
            terrain_mode="plane",
            seed=seed,
            actor_observation_noise=True,
            enable_pushes=False,
        )
        cfg = ppo_train_cfg()
        cfg["seed"] = seed
        cfg["runner"]["max_iterations"] = iterations
        cfg["runner"]["experiment_name"] = f"stage1_1_{mode}"
        runner = OnPolicyRunner(env, cfg, log_dir=str(run_dir), device=device)
        manifest["state"] = "training"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        runner.learn(iterations, init_at_random_ep_len=True)
        checkpoint = run_dir / f"model_{iterations}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing final checkpoint: {checkpoint}")
        manifest.update(
            {
                "state": "completed",
                "checkpoint_created": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "completed_at_unix_s": time.time(),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return run_dir
    except Exception as error:
        manifest.update({"state": "failed", "error": repr(error), "updated_at_unix_s": time.time()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        if env is not None:
            env.close()


def evaluate(
    mode: str,
    checkpoint: Path,
    *,
    num_envs: int,
    steps: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    spec = diagnostic_spec(mode)
    env = Stage11DiagnosticEnv(
        diagnostic=spec,
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
        env.set_training_iteration(500)
        cfg = ppo_train_cfg()
        cfg["seed"] = seed
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), load_optimizer=False)
        actor = runner.alg.actor_critic
        actor.eval()
        obs = env.get_observations()
        episode_length = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        episode_energy = torch.zeros(num_envs, device=env.device)
        completed_energy = []
        completed_length = []
        done_count = 0
        timeout_count = 0
        sums = {
            "velocity_squared_error": 0.0,
            "raw_action_abs": 0.0,
            "effective_action_abs": 0.0,
            "target_min": float("inf"),
            "target_max": float("-inf"),
            "saturation": 0.0,
            "electrical_j": 0.0,
            "mechanical_j": 0.0,
            "potential_j": 0.0,
        }
        started = time.time()
        with torch.inference_mode():
            for step_index in range(steps):
                base_velocity = quat_rotate_inverse(
                    env.root_states[:, 3:7], env.root_states[:, 7:10]
                )
                sums["velocity_squared_error"] += float(
                    (base_velocity[:, 0] - 1.0).square().sum().item()
                )
                raw_actions = actor.act_inference(obs)
                effective_actions = raw_actions
                if spec.action_bound is not None:
                    effective_actions = torch.clamp(
                        raw_actions, -spec.action_bound, spec.action_bound
                    )
                targets = env.default_dof_pos + env.cfg.action.scale_rad * torch.clamp(
                    effective_actions, -env.cfg.action.clip, env.cfg.action.clip
                )
                obs, _, _, dones, extras = env.step(raw_actions)
                sums["raw_action_abs"] += float(raw_actions.abs().sum().item())
                sums["effective_action_abs"] += float(effective_actions.abs().sum().item())
                sums["target_min"] = min(sums["target_min"], float(targets.min().item()))
                sums["target_max"] = max(sums["target_max"], float(targets.max().item()))
                sums["saturation"] += float(
                    extras["diagnostic"]["torque_saturation_ratio"].sum().item()
                )
                step_energy = env._electrical_energy + env._mechanical_energy + env._potential_energy
                episode_energy += step_energy
                episode_length += 1
                sums["electrical_j"] += float(env._electrical_energy.sum().item())
                sums["mechanical_j"] += float(env._mechanical_energy.sum().item())
                sums["potential_j"] += float(env._potential_energy.sum().item())
                if torch.any(dones):
                    completed_energy.append(episode_energy[dones].cpu())
                    completed_length.append(episode_length[dones].float().cpu())
                    done_count += int(dones.sum().item())
                    timeout_count += int(extras["time_outs"][dones].sum().item())
                    episode_energy[dones] = 0.0
                    episode_length[dones] = 0
                if (step_index + 1) % 500 == 0 or step_index + 1 == steps:
                    print(f"eval mode={mode} step={step_index + 1}/{steps}", flush=True)
        sample_count = num_envs * steps
        action_count = sample_count * env.num_actions
        duration_s = steps * env.policy_dt
        energy = torch.cat(completed_energy) if completed_energy else torch.empty(0)
        lengths = torch.cat(completed_length) if completed_length else torch.empty(0)
        return {
            "schema": "pace_stage1.stage1_1_diagnostic_eval.v1",
            "classification": "POST-TRAINING DIAGNOSTIC / NOT A BASELINE / NOT ECO",
            "mode": mode,
            "diagnostic": asdict(spec),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "fixed_command_vx_m_s": 1.0,
            "terrain": "plane",
            "seed": seed,
            "num_envs": num_envs,
            "policy_steps": steps,
            "duration_per_env_s": duration_s,
            "success_rate_timeout": timeout_count / done_count if done_count else None,
            "completed_episodes": done_count,
            "velocity_rmse_m_s": (sums["velocity_squared_error"] / sample_count) ** 0.5,
            "raw_action_mean_abs": sums["raw_action_abs"] / action_count,
            "effective_action_mean_abs": sums["effective_action_abs"] / action_count,
            "q_target_range_rad": [sums["target_min"], sums["target_max"]],
            "torque_saturation_ratio": sums["saturation"] / sample_count,
            "completed_episode_total_energy_j": _summary(energy) if energy.numel() else None,
            "completed_episode_length_steps": _summary(lengths) if lengths.numel() else None,
            "power_population_time_average_w": {
                "electrical": sums["electrical_j"] / (num_envs * duration_s),
                "mechanical_no_regen": sums["mechanical_j"] / (num_envs * duration_s),
                "potential": sums["potential_j"] / (num_envs * duration_s),
                "total": (
                    sums["electrical_j"] + sums["mechanical_j"] + sums["potential_j"]
                )
                / (num_envs * duration_s),
            },
            "wall_time_s": time.time() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-num-envs", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=2001)
    args = parser.parse_args()
    run_dir = train(args.mode, args.iterations, args.num_envs, args.seed, args.device)
    checkpoint = run_dir / f"model_{args.iterations}.pt"
    result = evaluate(
        args.mode,
        checkpoint,
        num_envs=args.eval_num_envs,
        steps=args.eval_steps,
        seed=args.seed,
        device=args.device,
    )
    output = run_dir / "gpt-固定速度确定性评估.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"EVAL={output}", flush=True)


if __name__ == "__main__":
    main()
