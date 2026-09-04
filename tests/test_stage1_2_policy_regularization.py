import unittest

from pace_stage1.stage1_2_metrics import (
    SCREENING_COEFFICIENTS,
    coefficient_label,
    raw_action_l2_penalty,
    task_success_mask,
)
import torch


class Stage12PolicyRegularizationTest(unittest.TestCase):
    def test_screening_matrix_is_exact(self):
        self.assertEqual(
            SCREENING_COEFFICIENTS,
            (0.0, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4),
        )
        self.assertEqual(coefficient_label(3.0e-5), "A2_lambda_3e-5")

    def test_raw_action_penalty_uses_pre_clip_per_joint_mean(self):
        raw = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        actual = raw_action_l2_penalty(raw, coefficient=1.0e-4)
        expected = torch.tensor([5.0e-4, 10.0e-4])
        torch.testing.assert_close(actual, expected)

    def test_task_success_requires_timeout_and_tracking(self):
        survived = torch.tensor([True, True, False])
        squared_error_sum = torch.tensor([4.0, 9.0, 1.0])
        lengths = torch.tensor([100, 100, 100])
        self.assertEqual(
            task_success_mask(survived, squared_error_sum, lengths).tolist(),
            [True, False, False],
        )


if __name__ == "__main__":
    unittest.main()
