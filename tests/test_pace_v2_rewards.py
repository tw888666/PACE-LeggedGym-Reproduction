import math
import unittest
from dataclasses import replace

import torch

from pace_stage1.config import STAGE1_CONFIG
from pace_stage1.pace_v2_rewards import pace_v2_environment_config
from pace_stage1.semantics import compute_task_reward_terms


class PaceV2RewardTest(unittest.TestCase):
    def reward(self, config=None, error=0., collision=False, touchdown=0., iteration=0):
        config = pace_v2_environment_config() if config is None else config
        return compute_task_reward_terms(
            torch.tensor([[error, 0., error, 0.]]), torch.zeros(1,3), torch.zeros(1,3),
            torch.tensor([collision]), torch.tensor([[touchdown,0.,0.,0.]]),
            torch.tensor([False]), torch.tensor([False]), iteration=iteration, config=config,
        )

    def test_sigma_is_the_denominator_not_its_square(self):
        terms = self.reward(error=.5)
        self.assertAlmostEqual(terms.velocity_tracking.item(), 2*math.exp(-1), places=6)
        self.assertAlmostEqual(terms.total.item(), .004*math.exp(-1), places=8)

    def test_collision_retains_negative_reward_and_legacy_stays_unchanged(self):
        terms = self.reward(collision=True)
        self.assertAlmostEqual(terms.total.item(), -.006, places=8)
        self.assertTrue(torch.equal(terms.task_before_clip, terms.task_after_clip))
        self.assertEqual(self.reward(config=STAGE1_CONFIG, collision=True).total.item(), 0.)
        self.assertTrue(STAGE1_CONFIG.rewards.only_positive_rewards)
        self.assertEqual(pace_v2_environment_config().rewards.reward_dt_s, .01)
        self.assertIs(pace_v2_environment_config().action, STAGE1_CONFIG.action)
        self.assertIs(pace_v2_environment_config().observation, STAGE1_CONFIG.observation)

    def test_ftd_half_life_and_uniform_dt_scaling(self):
        initial = self.reward(touchdown=1., iteration=0)
        half = self.reward(touchdown=1., iteration=500)
        self.assertAlmostEqual(initial.total.item(), .004, places=8)
        self.assertAlmostEqual(half.scaled_foot_touchdown.item(), -.0005, places=8)
        config = pace_v2_environment_config()
        doubled = replace(config, rewards=replace(config.rewards, reward_dt_s=.02))
        first = self.reward(collision=True, touchdown=1., iteration=500)
        second = self.reward(config=doubled, collision=True, touchdown=1., iteration=500)
        for name in ('scaled_velocity_tracking','scaled_collision','scaled_foot_touchdown','total'):
            torch.testing.assert_close(getattr(second,name), 2*getattr(first,name))


if __name__ == '__main__':
    unittest.main()
