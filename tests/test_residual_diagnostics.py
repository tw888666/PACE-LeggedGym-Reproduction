import unittest

import numpy as np

from pace_stage0.decoder import decode_fit
from pace_stage0.residual_diagnostics import (
    _bounds_block_audit,
    _first_sustained_crossing,
    _window_metrics,
)


class ResidualDiagnosticsTest(unittest.TestCase):
    def test_legacy_bounds_blocks(self):
        audit = _bounds_block_audit(decode_fit())
        self.assertEqual(audit["0:12"]["unique_bound_pairs"].tolist(), [[0.0, 0.5]])
        self.assertEqual(audit["12:24"]["unique_bound_pairs"].tolist(), [[0.0, 6.0]])
        self.assertEqual(audit["24:36"]["unique_bound_pairs"].tolist(), [[0.0, 0.5]])
        self.assertEqual(audit["36:48"]["unique_bound_pairs"].tolist(), [[-0.1, 0.1]])
        self.assertEqual(audit["48"]["unique_bound_pairs"].tolist(), [[0.0, 7.0]])

    def test_window_metrics(self):
        reference = np.zeros((600, 12), dtype=np.float64)
        ours = np.ones_like(reference)
        result = _window_metrics(ours, reference)
        self.assertEqual(tuple(result), ("first_10", "first_50", "first_100", "first_500", "full"))
        self.assertEqual(result["full"]["overall_rmse"], 1.0)
        self.assertTrue(all(value == 1.0 for value in result["full"]["per_joint_rmse"].values()))

    def test_first_sustained_crossing(self):
        values = np.asarray([0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.assertEqual(_first_sustained_crossing(values, 1.0), 4)
        self.assertIsNone(_first_sustained_crossing(values, 3.0))


if __name__ == "__main__":
    unittest.main()
