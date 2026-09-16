import math
import unittest

import torch

from pace_stage0.actuator import PACEActuatorCore
from pace_stage1.config import STAGE1_CONFIG
from pace_stage1.pace_v2_control import PaceV2ControlMixin, pace_v2_hard_limit_safe_target


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

    def test_five_degree_band_preserves_thirty_degree_inward_target_exactly(self):
        q_true = torch.tensor([1.0 - math.radians(1), -1.0 + math.radians(1)])
        target = torch.tensor([1.0 - math.radians(30), -1.0 + math.radians(30)])
        actual = pace_v2_hard_limit_safe_target(
            q_true, target, self.lower.expand(2), self.upper.expand(2), math.radians(5)
        )
        self.assertTrue(torch.equal(actual, target))

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


class PaceV2ControlWiringTest(unittest.TestCase):
    def make_control(self, bias=0.0, delay=0):
        control = PaceV2ControlMixin()
        control.cfg = STAGE1_CONFIG
        control.default_dof_pos = torch.tensor(STAGE1_CONFIG.action.default_joint_pose_rad).repeat(2, 1)
        control.actions = torch.full((2, 12), 4.0)
        control.lower_limits = torch.full((12,), -1.0)
        control.upper_limits = torch.full((12,), 1.0)
        control.dof_pos = torch.zeros(2, 12)
        control.dof_vel = torch.zeros(2, 12)
        control.actuator = PACEActuatorCore(torch.full((12,), bias), delay)
        # No legacy adapter is supplied: the v2 path must not depend on one.
        return control

    def test_raw_target_is_held_but_eq9_uses_latest_true_state_each_substep(self):
        control = self.make_control()
        target = control._compute_policy_target()
        expected = control.default_dof_pos + 2.0
        torch.testing.assert_close(target, expected)
        first = control._step_actuator(target)
        torch.testing.assert_close(first.raw_pd_torque, expected * control.actuator.kp)

        control.dof_pos[0] = 1.0 - math.radians(2.5)
        control.dof_pos[1] = 1.0
        second = control._step_actuator(target)
        safe = torch.stack(((expected[0] + 1.0) / 2.0, torch.ones(12)))
        torch.testing.assert_close(
            second.raw_pd_torque, control.actuator.kp * (safe - control.dof_pos)
        )
        self.assertTrue(torch.equal(target, expected))

    def test_true_state_precedes_bias_pd_envelope_and_delay(self):
        control = self.make_control(bias=0.2, delay=1)
        control.dof_pos.fill_(1.0)
        control.dof_vel.fill_(5.0)
        target = control._compute_policy_target()
        first = control._step_actuator(target)
        # q_encoder=0.8 is outside the band; using it in Eq9 would leave the
        # outward target unchanged. With q_true=1, the safe target must be 1.
        torch.testing.assert_close(first.q_encoder, torch.full((2, 12), 0.8))
        expected_pd = torch.full((2, 12), 0.2 * control.actuator.kp - 5.0 * control.actuator.kd)
        torch.testing.assert_close(first.raw_pd_torque, expected_pd)
        torch.testing.assert_close(first.applied_torque, torch.zeros(2, 12))

        control.actions.fill_(-100.0)
        second = control._step_actuator(control._compute_policy_target())
        # A large inward target remains unrestricted by Eq9, is then saturated
        # by the actuator, and only that saturated result enters the FIFO.
        self.assertTrue(torch.all(second.raw_pd_torque < -control.actuator.effort_limit))
        torch.testing.assert_close(second.applied_torque, first.saturated_torque)
        third = control._step_actuator(control._compute_policy_target())
        torch.testing.assert_close(third.applied_torque, second.saturated_torque)


if __name__ == "__main__":
    unittest.main()
