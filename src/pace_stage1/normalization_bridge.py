"""Input normalization for the unchanged rsl_rl 1.0.2 PPO/runner machinery."""

from __future__ import annotations

import importlib.metadata

import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner

from .empirical_normalization import (
    ActorCriticEmpiricalNormalizers,
    OBSERVATION_NORMALIZATION_SPEC,
)
from .protocol_gate import require_validation_training_allowed


class ObservationStatisticsEnv:
    """Update from each returned batch once, while returning unnormalized tensors."""

    def __init__(self, env, normalizers):
        self.env = env
        self.normalizers = normalizers
        self._actor_counted = False
        self._critic_counted = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        # The legacy runner assigns this when randomizing initial episode ages.
        self.env.episode_length_buf = value

    def get_observations(self):
        obs = self.env.get_observations()
        if not self._actor_counted and self.normalizers.actor.training:
            self.normalizers.actor.update(obs)
            self._actor_counted = True
        return obs

    def get_privileged_observations(self):
        obs = self.env.get_privileged_observations()
        if not self._critic_counted and self.normalizers.critic.training:
            self.normalizers.critic.update(obs)
            self._critic_counted = True
        return obs

    def step(self, actions):
        obs, critic_obs, rewards, dones, infos = self.env.step(actions)
        self.normalizers.actor.update(obs)
        self.normalizers.critic.update(critic_obs)
        self._actor_counted = self.normalizers.actor.training
        self._critic_counted = self.normalizers.critic.training
        return obs, critic_obs, rewards, dones, infos

    def reset(self):
        result = self.env.reset()
        self._actor_counted = False
        self._critic_counted = False
        return result


class NormalizationRunnerMixin:
    """Only install input modules and checkpoint semantics; inherit PPO and learn."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if importlib.metadata.version("rsl-rl") != "1.0.2":
            raise RuntimeError("normalization compatibility requires rsl_rl 1.0.2 PPO")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.normalizers = ActorCriticEmpiricalNormalizers(
            env.num_obs, env.num_privileged_obs
        ).to(device)
        # Only buffers are added. All optimizer parameter objects remain unchanged.
        model = self.alg.actor_critic
        model.actor = nn.Sequential(self.normalizers.actor, model.actor)
        model.critic = nn.Sequential(self.normalizers.critic, model.critic)
        self.env = ObservationStatisticsEnv(env, self.normalizers)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # v1.0.2 activates train mode after its initial observation getters.
        # Our initial batch must already see training-mode statistics updates.
        self.alg.actor_critic.train()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def _normalization_metadata(self):
        return {
            "spec": OBSERVATION_NORMALIZATION_SPEC,
            "actor": self.normalizers._normalizer_metadata(self.normalizers.actor),
            "critic": self.normalizers._normalizer_metadata(self.normalizers.critic),
        }

    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "observation_normalization": self._normalization_metadata(),
        }, path)

    def load(self, path, load_optimizer=True):
        payload = torch.load(path, map_location=self.device)
        if payload.get("observation_normalization") != self._normalization_metadata():
            raise RuntimeError("checkpoint is missing or has incompatible normalization semantics")
        state = payload["model_state_dict"]
        required = {prefix + name for prefix in ("actor.0.", "critic.0.")
                    for name in ("_mean", "_var", "_std", "count")}
        if not required.issubset(state):
            raise RuntimeError("checkpoint is missing normalizer statistics")
        self.alg.actor_critic.load_state_dict(state, strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
            # The adaptive v1.0.2 PPO also keeps the LR outside the optimizer.
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = payload["iter"]
        return payload.get("infos")


class PaceV2ValidationRunner(NormalizationRunnerMixin, OnPolicyRunner):
    """Production entry: construction requires explicit validation approval."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        require_validation_training_allowed()
        if getattr(getattr(env, "protocol_profile", None), "identity", None) != "PACE_V2_VALIDATION":
            raise ValueError("PACE v2 runner requires the PACE v2 validation environment")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
