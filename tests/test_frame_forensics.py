from __future__ import annotations

import unittest

import numpy as np

from pace_stage0.frame_forensics import _relationship_stats


class FrameForensicsTest(unittest.TestCase):
    def test_relationship_stats_and_contributions(self):
        reference = np.zeros((2, 2), dtype=np.float64)
        sim = np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
        result = _relationship_stats(sim, reference, ("J0", "J1"))

        self.assertAlmostEqual(result["overall"]["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(result["overall"]["state0_rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(
            result["per_joint"][0]["squared_error_contribution_ratio"], 0.2
        )
        self.assertAlmostEqual(
            result["per_joint"][1]["squared_error_contribution_ratio"], 0.8
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            _relationship_stats(np.zeros((2, 2)), np.zeros((2, 3)), ("J0", "J1"))


if __name__ == "__main__":
    unittest.main()
