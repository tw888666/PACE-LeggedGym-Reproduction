import math
import unittest

import torch

from pace_stage0.target_adapter import LocomotionTargetAdapter


class TestLocomotionTargetAdapter(unittest.TestCase):
    def test_scale_offset_and_absolute_output(self):
        adapter = LocomotionTargetAdapter()
        action = torch.tensor([1.0, -1.0, 0.0])
        q0 = torch.tensor([0.2, -0.3, 0.4])
        lower = torch.full((3,), -2.0)
        upper = torch.full((3,), 2.0)
        actual = adapter(action, q0, lower, upper)
        torch.testing.assert_close(actual, torch.tensor([0.7, -0.8, 0.4]))

    def test_five_degree_inward_band(self):
        adapter = LocomotionTargetAdapter()
        lower = torch.tensor([-1.0, -1.0])
        upper = torch.tensor([1.0, 1.0])
        actual = adapter(torch.tensor([-100.0, 100.0]), torch.zeros(2), lower, upper)
        band = math.radians(5.0)
        torch.testing.assert_close(actual, torch.tensor([-1.0 + band, 1.0 - band]))

    def test_adapter_has_no_actuator_state(self):
        adapter = LocomotionTargetAdapter()
        self.assertFalse(hasattr(adapter, "encoder_bias"))
        self.assertFalse(hasattr(adapter, "kp"))
        self.assertFalse(hasattr(adapter, "delay_steps"))


if __name__ == "__main__":
    unittest.main()

