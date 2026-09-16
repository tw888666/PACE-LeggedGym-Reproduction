"""PACE-v2 compatibility with the preregistered RSL-RL 3.0.1 normalizer."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ._rsl_rl_normalization_v3_0_1 import EmpiricalNormalization


OBSERVATION_NORMALIZATION_SPEC = "OBSERVATION_NORMALIZATION_RECONSTRUCTION_V1"


class RunningMeanVarianceNormalizer(EmpiricalNormalization):
    """Dimension-aware facade; arithmetic and buffers come from upstream 3.0.1."""

    def __init__(self, dimension: int, epsilon: float = 1.0e-2):
        if dimension <= 0 or epsilon <= 0:
            raise ValueError("dimension and epsilon must be positive")
        super().__init__(dimension, eps=epsilon, until=None)
        self.dimension = int(dimension)
        self.epsilon = float(epsilon)

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise ValueError("observations must have shape [batch, dimension]")
        if observations.shape[0] == 0:
            return
        super().update(observations.detach().to(self._mean))

    @property
    def running_mean(self):
        return self._mean.squeeze(0)

    @property
    def running_variance(self):
        return self._var.squeeze(0)

    @property
    def sample_count(self):
        return self.count


class ActorCriticEmpiricalNormalizers(nn.Module):
    """Keep actor and critic running statistics independent and checkpointable."""

    CHECKPOINT_KEY = "pace_v2_empirical_normalizers"

    def __init__(
        self,
        actor_dimension: int,
        critic_dimension: int,
        epsilon: float = 1.0e-2,
    ):
        super().__init__()
        self.actor = RunningMeanVarianceNormalizer(actor_dimension, epsilon)
        self.critic = RunningMeanVarianceNormalizer(critic_dimension, epsilon)

    def normalize_actor(
        self, observations: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        if update:
            self.actor.update(observations)
        return self.actor(observations)

    def normalize_critic(
        self, observations: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        if update:
            self.critic.update(observations)
        return self.critic(observations)

    def add_to_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        if self.CHECKPOINT_KEY in checkpoint:
            raise RuntimeError("checkpoint already contains empirical normalizer state")
        checkpoint[self.CHECKPOINT_KEY] = {
            "schema": "pace_v2_empirical_normalizers.rsl301.v1",
            "actor": self._normalizer_metadata(self.actor),
            "critic": self._normalizer_metadata(self.critic),
            "state_dict": {
                key: value.detach().clone() for key, value in self.state_dict().items()
            },
        }

    def load_from_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        if self.CHECKPOINT_KEY not in checkpoint:
            raise RuntimeError("checkpoint is missing empirical normalizer state")
        payload = checkpoint[self.CHECKPOINT_KEY]
        if not isinstance(payload, dict) or payload.get("schema") != (
            "pace_v2_empirical_normalizers.rsl301.v1"
        ):
            raise RuntimeError("unsupported empirical normalizer checkpoint state")
        if payload.get("actor") != self._normalizer_metadata(self.actor):
            raise RuntimeError("checkpoint actor normalizer config differs from runtime")
        if payload.get("critic") != self._normalizer_metadata(self.critic):
            raise RuntimeError("checkpoint critic normalizer config differs from runtime")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("checkpoint empirical normalizer tensors are missing")
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _normalizer_metadata(
        normalizer: RunningMeanVarianceNormalizer,
    ) -> Dict[str, object]:
        return {
            "dimension": normalizer.dimension,
            "epsilon": normalizer.epsilon,
            "output_clip": None,
            "until": None,
            "spec": OBSERVATION_NORMALIZATION_SPEC,
        }
