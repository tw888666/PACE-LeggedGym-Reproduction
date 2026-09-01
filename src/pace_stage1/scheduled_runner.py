"""rsl_rl v1.0.2 runner with the frozen PACE iteration schedules."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from .config import STAGE1_CONFIG, Stage1Config
from .runner_bridge import apply_iteration_schedules


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


class Stage1ScheduledOnPolicyRunner(OnPolicyRunner):
    """Reference runner with one deliberate change: apply frozen schedules per rollout."""

    def __init__(
        self,
        *args,
        stage1_config: Stage1Config = STAGE1_CONFIG,
        runtime_manifest_path: Optional[Path] = None,
        runtime_manifest: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.stage1_config = stage1_config
        self.runtime_manifest_path = runtime_manifest_path
        self.runtime_manifest = runtime_manifest
        super().__init__(*args, **kwargs)

    def _record_progress(self, state: str, iteration: int, entropy: float) -> None:
        if self.runtime_manifest_path is None or self.runtime_manifest is None:
            return
        self.runtime_manifest["state"] = state
        self.runtime_manifest["current_iteration"] = int(iteration)
        self.runtime_manifest["entropy_coefficient"] = float(entropy)
        self.runtime_manifest["updated_at_unix_s"] = time.time()
        _atomic_json(self.runtime_manifest_path, self.runtime_manifest)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Run reference rsl_rl v1.0.2 PPO with pre-rollout schedule updates."""
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_iterations = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, total_iterations):
            entropy = apply_iteration_schedules(
                self.alg, self.env, it, self.stage1_config
            )
            self._record_progress("collecting_rollout", it, entropy)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                collection_time = time.time() - start
                update_start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learn_time = time.time() - update_start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                self._record_progress("checkpoint_saved", it, entropy)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            )
        )
        self._record_progress(
            "completed", self.current_learning_iteration, self.alg.entropy_coef
        )
