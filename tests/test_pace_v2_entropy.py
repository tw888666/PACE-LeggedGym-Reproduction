import unittest

from pace_stage1.pace_v2_entropy import (
    PaceV2EntropyScheduler,
    pace_v2_entropy_coefficient,
)
from pace_stage1.pace_v2_profiles import PaceV2EntropyConfig


class PaceV2EntropyScheduleTest(unittest.TestCase):
    def setUp(self):
        self.config = PaceV2EntropyConfig(
            initial=0.002,
            final=0.0005,
            turnover_iteration=20_000,
            slope_eta=0.001,
        )

    def test_unresolved_eta_refuses_evaluation(self):
        with self.assertRaises(RuntimeError):
            pace_v2_entropy_coefficient(0, PaceV2EntropyConfig(slope_eta=None))

    def test_initial_turnover_final_and_monotonicity(self):
        at_start = pace_v2_entropy_coefficient(0, self.config)
        at_turnover = pace_v2_entropy_coefficient(20_000, self.config)
        at_end = pace_v2_entropy_coefficient(40_000, self.config)
        self.assertAlmostEqual(at_start, self.config.initial, places=10)
        self.assertAlmostEqual(
            at_turnover, (self.config.initial + self.config.final) / 2.0
        )
        self.assertAlmostEqual(at_end, self.config.final, places=10)
        values = [
            pace_v2_entropy_coefficient(iteration, self.config)
            for iteration in range(0, 40_001, 1_000)
        ]
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_scheduler_applies_and_round_trips_state(self):
        class Algorithm:
            entropy_coef = None

        algorithm = Algorithm()
        scheduler = PaceV2EntropyScheduler(self.config)
        expected = scheduler.apply(algorithm, 12_345)
        self.assertEqual(algorithm.entropy_coef, expected)

        restored = PaceV2EntropyScheduler(self.config)
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.last_iteration, 12_345)

    def test_checkpoint_config_mismatch_fails_closed(self):
        scheduler = PaceV2EntropyScheduler(self.config)
        state = scheduler.state_dict()
        other = PaceV2EntropyScheduler(
            PaceV2EntropyConfig(slope_eta=0.002)
        )
        with self.assertRaises(RuntimeError):
            other.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
