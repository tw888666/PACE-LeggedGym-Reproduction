"""Separate actor/critic empirical normalizers with checkpointable statistics."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class RunningMeanVarianceNormalizer(nn.Module):
    """Normalize the final tensor dimension using population running moments."""

    def __init__(self, dimension: int, epsilon: float = 1.0e-8, clip: float = 100.0):
        super().__init__()
        if dimension <= 0:
            raise ValueError("normalizer dimension must be positive")
        if epsilon <= 0 or clip <= 0:
            raise ValueError("normalizer epsilon and clip must be positive")
        self.dimension = int(dimension)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.register_buffer("running_mean", torch.zeros(self.dimension))
        self.register_buffer("running_variance", torch.ones(self.dimension))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        self._validate_observations(observations)
        flattened = observations.detach().reshape(-1, self.dimension)
        if flattened.shape[0] == 0:
            return
        batch = flattened.to(
            device=self.running_mean.device, dtype=self.running_mean.dtype
        )
        batch_count = float(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_variance = batch.var(dim=0, unbiased=False)
        old_count = float(self.sample_count.item())
        if old_count == 0.0:
            self.running_mean.copy_(batch_mean)
            self.running_variance.copy_(batch_variance)
            self.sample_count.fill_(batch_count)
            return

        total_count = old_count + batch_count
        delta = batch_mean - self.running_mean
        combined_mean = self.running_mean + delta * (batch_count / total_count)
        old_second_moment = self.running_variance * old_count
        batch_second_moment = batch_variance * batch_count
        correction = delta.square() * (old_count * batch_count / total_count)
        combined_variance = (
            old_second_moment + batch_second_moment + correction
        ) / total_count
        self.running_mean.copy_(combined_mean)
        self.running_variance.copy_(combined_variance)
        self.sample_count.fill_(total_count)

    def forward(self, observations: torch.Tensor, update: bool = False) -> torch.Tensor:
        self._validate_observations(observations)
        if update:
            self.update(observations)
        mean = self.running_mean.to(device=observations.device, dtype=observations.dtype)
        variance = self.running_variance.to(
            device=observations.device, dtype=observations.dtype
        )
        normalized = (observations - mean) / torch.sqrt(variance + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if not torch.is_floating_point(observations):
            raise TypeError("normalizer observations must be floating-point tensors")
        if observations.ndim < 1 or observations.shape[-1] != self.dimension:
            raise ValueError(
                f"expected observation final dimension {self.dimension}, "
                f"got {tuple(observations.shape)}"
            )
        if not torch.isfinite(observations).all():
            raise ValueError("normalizer observations must be finite")


class ActorCriticEmpiricalNormalizers(nn.Module):
    """Keep actor and critic running statistics independent and checkpointable."""

    CHECKPOINT_KEY = "pace_v2_empirical_normalizers"

    def __init__(
        self,
        actor_dimension: int,
        critic_dimension: int,
        epsilon: float = 1.0e-8,
        clip: float = 100.0,
    ):
        super().__init__()
        self.actor = RunningMeanVarianceNormalizer(actor_dimension, epsilon, clip)
        self.critic = RunningMeanVarianceNormalizer(critic_dimension, epsilon, clip)

    def normalize_actor(
        self, observations: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        return self.actor(observations, update=update)

    def normalize_critic(
        self, observations: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        return self.critic(observations, update=update)

    def add_to_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        if self.CHECKPOINT_KEY in checkpoint:
            raise RuntimeError("checkpoint already contains empirical normalizer state")
        checkpoint[self.CHECKPOINT_KEY] = {
            "schema": "pace_v2_empirical_normalizers.v1",
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
            "pace_v2_empirical_normalizers.v1"
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
            "clip": normalizer.clip,
        }
