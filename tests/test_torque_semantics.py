import unittest

import numpy as np

from pace_stage0.constants import KD, KP
from pace_stage0.torque_semantics import (
    _align_dataset_candidate,
    _dcmotor_clip,
    _delay_trace,
    _lag_sweep,
    _metrics,
)


class TorqueSemanticsTest(unittest.TestCase):
    def test_delay_trace_has_three_zero_rows(self):
        value = np.arange(12, dtype=np.float64).reshape(6, 2)
        delayed = _delay_trace(value, 3)
        np.testing.assert_array_equal(delayed[:3], np.zeros((3, 2)))
        np.testing.assert_array_equal(delayed[3:], value[:-3])

    def test_lag_definition_dataset_k_candidate_k_plus_lag(self):
        candidate = np.arange(8, dtype=np.float64)[:, None]
        dataset = np.zeros_like(candidate)
        dataset[1:] = candidate[:-1]
        reference, aligned, dataset_indices, candidate_indices = (
            _align_dataset_candidate(dataset, candidate, -1)
        )
        np.testing.assert_array_equal(reference, aligned)
        np.testing.assert_array_equal(dataset_indices, np.arange(1, 8))
        np.testing.assert_array_equal(candidate_indices, np.arange(0, 7))
        self.assertEqual(_lag_sweep(dataset, candidate)["best"]["lag"], -1)

    def test_synthetic_full_pd_bias_delay_semantics(self):
        count = 40
        target = np.linspace(-0.2, 0.4, count)[:, None]
        position = np.linspace(-0.1, 0.2, count)[:, None]
        velocity = np.linspace(-1.0, 1.0, count)[:, None]
        bias = np.asarray([0.02])
        pre = KP * ((target - bias) - position) - KD * velocity
        delayed = _delay_trace(pre, 3)
        dataset = np.zeros_like(delayed)
        dataset[1:] = delayed[:-1]
        sweep = _lag_sweep(dataset, delayed)
        self.assertEqual(sweep["best"]["lag"], -1)
        self.assertEqual(sweep["best"]["rmse_Nm"], 0.0)

    def test_dcmotor_clip_is_identity_inside_envelope(self):
        torque = np.asarray([[1.0, -2.0], [3.0, -4.0]])
        velocity = np.zeros_like(torque)
        np.testing.assert_array_equal(_dcmotor_clip(torque, velocity), torque)

    def test_metrics_use_reference_minus_candidate_residual(self):
        candidate = np.asarray([[1.0], [2.0]])
        reference = np.asarray([[2.0], [4.0]])
        result = _metrics(candidate, reference)
        self.assertEqual(result["mean_residual_ref_minus_candidate_Nm"], 1.5)


if __name__ == "__main__":
    unittest.main()
