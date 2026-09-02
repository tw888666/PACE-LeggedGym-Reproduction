"""Bounded in-memory rsl_rl PPO smoke; it never writes a checkpoint."""

from __future__ import annotations

# Isaac Gym Preview 4 must be imported before torch/rsl_rl.
from .env import Stage1LocomotionEnv

import importlib.metadata
import json
import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from .config import STAGE1_CONFIG, ppo_train_cfg
from .runner_bridge import apply_task_iteration


def run() -> dict:
    smoke = STAGE1_CONFIG.ppo_smoke
    version = importlib.metadata.version("rsl-rl")
    if version != "1.0.2":
        raise RuntimeError(f"rsl_rl must be 1.0.2, got {version}")
    train = ppo_train_cfg()
    with Stage1LocomotionEnv(
        num_envs=smoke.num_envs,
        sim_device="cpu",
        headless=True,
        terrain_mode="plane",
        seed=smoke.seed,
        actor_observation_noise=False,
        enable_pushes=False,
    ) as env:
        actor_critic = ActorCritic(
            env.num_obs,
            env.num_privileged_obs,
            env.num_actions,
            **train["policy"],
        )
        algorithm = PPO(actor_critic, device="cpu", **train["algorithm"])
        algorithm.init_storage(
            env.num_envs,
            smoke.rollout_steps,
            [env.num_obs],
            [env.num_privileged_obs],
            [env.num_actions],
        )
        obs = env.get_observations()
        critic_obs = env.get_privileged_observations()
        apply_task_iteration(env, 0)
        for _ in range(smoke.rollout_steps):
            with torch.inference_mode():
                actions = algorithm.act(obs, critic_obs)
                obs, critic_obs, rewards, dones, infos = env.step(actions)
                algorithm.process_env_step(rewards, dones, infos)
        with torch.inference_mode():
            algorithm.compute_returns(critic_obs)
        value_loss, surrogate_loss = algorithm.update()
        finite = all(
            math.isfinite(float(value))
            for value in (value_loss, surrogate_loss, rewards.mean().item())
        )
    return {
        "classification": smoke.classification,
        "rsl_rl_version": version,
        "num_envs": smoke.num_envs,
        "rollout_steps": smoke.rollout_steps,
        "optimizer_updates": smoke.optimizer_updates,
        "formal_seed": smoke.formal_seed,
        "checkpoint_created": smoke.checkpoint_created,
        "formal_ablation_PPO_started": False,
        "in_memory_PPO_update_completed": True,
        "value_loss": float(value_loss),
        "surrogate_loss": float(surrogate_loss),
        "finite": finite,
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
