import unittest

import numpy as np

from pace_stage0.accumulation_audit import (
    _derivative_candidates,
    _indexed_lag_metrics,
    _simulation_cases,
    _target_quantization,
    _time_audit,
    _velocity_semantics,
)


class AccumulationAuditUtilitiesTest(unittest.TestCase):
    def test_derivative_hypothesis_recovers_linear_velocity(self):
        count = 40
        time = np.arange(count, dtype=np.float64) * 0.0025
        slopes = np.arange(1, 13, dtype=np.float64)
        position = time[:, None] * slopes[None, :]
        velocity = np.broadcast_to(slopes, position.shape).copy()
        audit = _velocity_semantics(position, velocity, time)
        self.assertLess(audit["best_rmse"], 1e-12)
        self.assertGreater(audit["best_correlation"], 0.999999)

    def test_lag_convention(self):
        reference = np.arange(30, dtype=np.float64)[:, None]
        candidate = reference[2:20].copy()
        indices = np.arange(0, 18)
        result = _indexed_lag_metrics(candidate, indices, reference, lag=2)
        self.assertEqual(result["rmse"], 0.0)

    def test_time_audit_detects_jitter(self):
        time = np.asarray([0.0, 0.0025, 0.0051, 0.0075], dtype=np.float32)
        audit = _time_audit(time)
        self.assertFalse(audit["strictly_equidistant"])
        self.assertGreater(audit["unique_diff_count"], 1)

    def test_float32_target_closes_precision_branch(self):
        target = np.arange(120, dtype=np.float32).reshape(10, 12) / 10
        audit = _target_quantization(target)
        self.assertTrue(audit["source_is_float32"])
        self.assertEqual(audit["source_to_float32_max_abs_error"], 0.0)
        self.assertTrue(audit["precision_branch"].startswith("CLOSED"))

    def test_simulation_matrix_has_unique_cases_and_frozen_shifts(self):
        cases = _simulation_cases()
        names = [case.name for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(next(c for c in cases if c.name == "formal_zero").target_shift, 0)
        self.assertEqual(next(c for c in cases if c.name == "target_t_minus_1").target_shift, -1)
        self.assertEqual(next(c for c in cases if c.name == "target_t_plus_1").target_shift, 1)
        one_step = next(c for c in cases if c.name == "one_step_author_initial")
        self.assertTrue(one_step.author_initial_state)
        self.assertEqual(one_step.reset_horizon, 1)


if __name__ == "__main__":
    unittest.main()
