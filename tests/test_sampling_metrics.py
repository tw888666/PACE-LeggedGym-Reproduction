import unittest

import numpy as np

from pace_stage0.metrics import lag_rmse, per_joint_rmse, rmse
from pace_stage0.sampling import run_indexed_replay


class TestSamplingSemantics(unittest.TestCase):
    def test_state_t_target_t_state_t_plus_one(self):
        targets = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)

        def transition(t, target, q, qdot):
            return target.copy(), np.asarray([t + 1.0], dtype=np.float32), {"t": t}

        replay = run_indexed_replay(
            targets,
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            transition,
        )
        np.testing.assert_array_equal(replay.q_true[:, 0], [1.0, 10.0, 20.0])
        np.testing.assert_array_equal(replay.qdot[:, 0], [0.0, 1.0, 2.0])
        self.assertEqual([r["t"] for r in replay.interval_records], [0, 1])


class TestMetrics(unittest.TestCase):
    def test_rmse_and_per_joint(self):
        a = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        b = np.asarray([[0.0, 0.0], [2.0, 2.0]])
        self.assertAlmostEqual(rmse(a, b), np.sqrt(0.5))
        np.testing.assert_allclose(per_joint_rmse(a, b), [0.0, 1.0])

    def test_lag_convention(self):
        reference = np.arange(6.0)[:, None]
        ours = reference[1:].copy()
        ours = np.concatenate([ours, [[99.0]]], axis=0)
        self.assertEqual(lag_rmse(ours, reference, 1), 0.0)


if __name__ == "__main__":
    unittest.main()

