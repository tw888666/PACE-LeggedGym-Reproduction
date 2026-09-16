import io
import unittest

import torch

from pace_stage1.empirical_normalization import (
    ActorCriticEmpiricalNormalizers,
    RunningMeanVarianceNormalizer,
)


class RunningMeanVarianceNormalizerTest(unittest.TestCase):
    def test_running_moments_and_normalized_values(self):
        normalizer = RunningMeanVarianceNormalizer(2)
        observations = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        normalizer.update(observations)
        actual = normalizer(observations)
        torch.testing.assert_close(normalizer.running_mean, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(normalizer.running_variance, torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(actual, torch.tensor([[-1/1.01, -2/2.01], [1/1.01, 2/2.01]]))
        self.assertEqual(normalizer.sample_count.item(), 2.0)

    def test_batched_updates_match_single_update(self):
        observations = torch.tensor(
            [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]]
        )
        whole = RunningMeanVarianceNormalizer(2)
        split = RunningMeanVarianceNormalizer(2)
        whole.update(observations)
        split.update(observations[:2])
        split.update(observations[2:])
        torch.testing.assert_close(split.running_mean, whole.running_mean)
        torch.testing.assert_close(split.running_variance, whole.running_variance)
        self.assertEqual(split.sample_count.item(), whole.sample_count.item())

    def test_actor_and_critic_statistics_are_independent_and_checkpointed(self):
        normalizers = ActorCriticEmpiricalNormalizers(2, 3)
        normalizers.normalize_actor(torch.tensor([[1.0, 3.0]]), update=True)
        normalizers.normalize_critic(torch.tensor([[2.0, 4.0, 8.0]]), update=True)
        checkpoint = {}
        normalizers.add_to_checkpoint(checkpoint)

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        restored_checkpoint = torch.load(buffer)
        restored = ActorCriticEmpiricalNormalizers(2, 3)
        restored.load_from_checkpoint(restored_checkpoint)

        torch.testing.assert_close(
            restored.actor.running_mean, normalizers.actor.running_mean
        )
        torch.testing.assert_close(
            restored.critic.running_mean, normalizers.critic.running_mean
        )
        self.assertEqual(restored.actor.sample_count.item(), 1.0)
        self.assertEqual(restored.critic.sample_count.item(), 1.0)

    def test_missing_checkpoint_state_fails_closed(self):
        with self.assertRaises(RuntimeError):
            ActorCriticEmpiricalNormalizers(2, 3).load_from_checkpoint({})

    def test_checkpoint_config_mismatch_fails_closed(self):
        source = ActorCriticEmpiricalNormalizers(2, 3, epsilon=0.01)
        checkpoint = {}
        source.add_to_checkpoint(checkpoint)
        with self.assertRaises(RuntimeError):
            ActorCriticEmpiricalNormalizers(2, 3, epsilon=0.02).load_from_checkpoint(
                checkpoint
            )


if __name__ == "__main__":
    unittest.main()
