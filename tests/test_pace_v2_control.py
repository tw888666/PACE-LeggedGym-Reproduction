import math
import unittest

import torch

from pace_stage1.pace_v2_control import pace_v2_hard_limit_safe_target


class PaceV2HardLimitSafeTargetTest(unittest.TestCase):
    def setUp(self):
        self.lower = torch.tensor([-1.0])
        self.upper = torch.tensor([1.0])
        self.band = 0.1

    def reshape(self, q_true, q_target):
        return pace_v2_hard_limit_safe_target(
            torch.tensor([q_true]),
            torch.tensor([q_target]),
            self.lower,
            self.upper,
            self.band,
        ).item()

    def test_target_is_unchanged_outside_soft_limit_bands(self):
        self.assertEqual(self.reshape(0.0, 20.0), 20.0)
        self.assertEqual(self.reshape(0.0, -20.0), -20.0)

    def test_target_at_hard_limit_produces_zero_outward_position_error(self):
        self.assertAlmostEqual(self.reshape(1.0, 20.0), 1.0)
        self.assertAlmostEqual(self.reshape(-1.0, -20.0), -1.0)

    def test_soft_band_interpolates_outward_target(self):
        self.assertAlmostEqual(self.reshape(0.95, 2.0), 1.5, places=6)
        self.assertAlmostEqual(self.reshape(-0.95, -2.0), -1.5, places=6)

    def test_target_away_from_limit_is_never_reshaped(self):
        self.assertEqual(self.reshape(0.95, -2.0), -2.0)
        self.assertEqual(self.reshape(-0.95, 2.0), 2.0)

    def test_mapping_is_continuous_at_soft_band_boundary(self):
        epsilon = 1.0e-6
        outside = self.reshape(0.9 - epsilon, 2.0)
        boundary = self.reshape(0.9, 2.0)
        inside = self.reshape(0.9 + epsilon, 2.0)
        self.assertAlmostEqual(outside, boundary, places=6)
        self.assertLess(abs(inside - boundary), 2.0e-5)

    def test_scalar_and_per_joint_soft_bands_are_supported(self):
        q_true = torch.tensor([[0.95, -0.95]])
        q_target = torch.tensor([[2.0, -2.0]])
        lower = torch.tensor([[-1.0, -1.0]])
        upper = torch.tensor([[1.0, 1.0]])
        actual = pace_v2_hard_limit_safe_target(
            q_true,
            q_target,
            lower,
            upper,
            torch.tensor([0.1, 0.1]),
        )
        torch.testing.assert_close(actual, torch.tensor([[1.5, -1.5]]))

    def test_invalid_limits_or_band_fail_closed(self):
        with self.assertRaises(ValueError):
            pace_v2_hard_limit_safe_target(
                torch.zeros(1), torch.zeros(1), self.upper, self.lower, self.band
            )
        with self.assertRaises(ValueError):
            pace_v2_hard_limit_safe_target(
                torch.zeros(1), torch.zeros(1), self.lower, self.upper, 0.0
            )
        with self.assertRaises(ValueError):
            pace_v2_hard_limit_safe_target(
                torch.zeros(1), torch.zeros(1), self.lower, self.upper, math.inf
            )


if __name__ == "__main__":
    unittest.main()
