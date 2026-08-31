import unittest

import torch

from pace_stage0.actuator import PACEActuatorCore, TorqueDelayLine


class TestPACEActuatorCore(unittest.TestCase):
    def test_encoder_frame_and_pd(self):
        core = PACEActuatorCore(torch.tensor([0.1, -0.2]), delay_steps=0)
        q_true = torch.tensor([0.3, 0.1])
        qdot = torch.tensor([2.0, -3.0])
        target = torch.tensor([0.2, 0.3])
        step = core.step(target, q_true, qdot)
        torch.testing.assert_close(step.q_encoder, torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(step.raw_pd_torque, torch.tensor([-1.2, 1.8]))
        torch.testing.assert_close(step.applied_torque, step.saturated_torque)

    def test_four_quadrant_envelope(self):
        core = PACEActuatorCore(torch.zeros(1), delay_steps=0)
        high = torch.tensor([1000.0])
        low = -high
        zero = torch.zeros(1)
        v_limit = torch.tensor([core.velocity_limit])
        v_corner = torch.tensor([core.velocity_corner])
        torch.testing.assert_close(core.saturate(high, zero), torch.tensor([89.0]))
        torch.testing.assert_close(core.saturate(low, zero), torch.tensor([-89.0]))
        torch.testing.assert_close(core.saturate(high, v_limit), torch.tensor([0.0]))
        torch.testing.assert_close(core.saturate(low, v_limit), torch.tensor([-89.0]))
        torch.testing.assert_close(core.saturate(high, v_corner), torch.tensor([-89.0]))
        torch.testing.assert_close(core.saturate(low, v_corner), torch.tensor([-89.0]))
        torch.testing.assert_close(core.saturate(high, 100.0 * v_corner), torch.tensor([-89.0]))

    def test_delay_is_applied_after_saturation(self):
        core = PACEActuatorCore(torch.zeros(1), delay_steps=3)
        core.reset(torch.zeros(1))
        outputs = []
        saturated = []
        for target in (10.0, -10.0, 10.0, -10.0):
            step = core.step(torch.tensor([target]), torch.zeros(1), torch.zeros(1))
            saturated.append(step.saturated_torque.item())
            outputs.append(step.applied_torque.item())
        self.assertEqual(saturated, [89.0, -89.0, 89.0, -89.0])
        self.assertEqual(outputs, [0.0, 0.0, 0.0, 89.0])

    def test_core_has_no_locomotion_preprocessing(self):
        core = PACEActuatorCore(torch.zeros(12), delay_steps=3)
        self.assertFalse(hasattr(core, "action_scale"))
        self.assertFalse(hasattr(core, "default_dof_pos"))
        self.assertFalse(hasattr(core, "soft_band"))


class TestTorqueDelayLine(unittest.TestCase):
    def test_reset_restores_zero_padding(self):
        line = TorqueDelayLine(2)
        reference = torch.zeros(1)
        line.reset(reference)
        self.assertEqual(line.compute(torch.tensor([1.0])).item(), 0.0)
        self.assertEqual(line.compute(torch.tensor([2.0])).item(), 0.0)
        self.assertEqual(line.compute(torch.tensor([3.0])).item(), 1.0)
        line.reset(reference)
        self.assertEqual(line.compute(torch.tensor([4.0])).item(), 0.0)


if __name__ == "__main__":
    unittest.main()

