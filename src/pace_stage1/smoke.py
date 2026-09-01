"""Bounded task-only MDP smoke; this module cannot create a PPO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json


def run(
    num_envs: int = 2,
    steps: int = 3,
    seed: int = 123,
    terrain_mode: str = "plane",
) -> dict:
    if num_envs > 8 or steps > 16:
        raise ValueError("SMOKE / NON-EXPERIMENTAL limits are num_envs<=8 and steps<=16")
    # Import lazily to preserve Isaac Gym's required import order.
    from .env import Stage1LocomotionEnv
    import torch

    result = {
        "classification": "SMOKE / NON-EXPERIMENTAL",
        "ppo_started": False,
        "checkpoint_created": False,
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "terrain_mode": terrain_mode,
    }
    with Stage1LocomotionEnv(
        num_envs=num_envs,
        sim_device="cpu",
        headless=True,
        terrain_mode=terrain_mode,
        seed=seed,
        actor_observation_noise=False,
        enable_pushes=False,
    ) as env:
        obs, critic = env.reset()
        result["reset_actor_shape"] = list(obs.shape)
        result["reset_critic_shape"] = list(critic.shape)
        result["reset_actor_sha256"] = hashlib.sha256(obs.cpu().numpy().tobytes()).hexdigest()
        result["reset_critic_sha256"] = hashlib.sha256(critic.cpu().numpy().tobytes()).hexdigest()
        rewards = None
        dones = None
        for _ in range(steps):
            obs, critic, rewards, dones, _ = env.step(
                torch.zeros((num_envs, env.num_actions), device=env.device)
            )
        result["step_actor_shape"] = list(obs.shape)
        result["step_critic_shape"] = list(critic.shape)
        result["reward_shape"] = list(rewards.shape)
        result["done_shape"] = list(dones.shape)
        result["final_actor_sha256"] = hashlib.sha256(obs.cpu().numpy().tobytes()).hexdigest()
        result["final_reward_sha256"] = hashlib.sha256(rewards.cpu().numpy().tobytes()).hexdigest()
        result["finite"] = bool(torch.isfinite(obs).all() and torch.isfinite(critic).all() and torch.isfinite(rewards).all())
        result["reward_term_names"] = sorted(env.extras["reward_terms"].keys())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--terrain-mode", choices=("plane", "trimesh"), default="plane")
    args = parser.parse_args()
    print(json.dumps(run(args.num_envs, args.steps, args.seed, args.terrain_mode), indent=2))


if __name__ == "__main__":
    main()
