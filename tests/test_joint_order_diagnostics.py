import unittest

import numpy as np

from pace_stage0.constants import CANONICAL_JOINT_NAMES, KP
from pace_stage0.data import load_replay_data
from pace_stage0.decoder import decode_fit
from pace_stage0.joint_order_diagnostics import (
    RAW_LEG_ORDER,
    _channel_order_audit,
    _rank_bias_leg_permutations,
    _torque_metrics,
)


class JointOrderDiagnosticsTest(unittest.TestCase):
    def test_bias_leg_permutation_ranking_recovers_raw(self):
        fit = decode_fit()
        raw_bias = fit.params_raw[36:48]
        exact_delta = KP * raw_bias
        ranking = _rank_bias_leg_permutations(exact_delta, raw_bias)
        self.assertEqual(tuple(ranking[0]["leg_order"]), RAW_LEG_ORDER)
        self.assertEqual(ranking[0]["rmse_Nm"], 0.0)
        self.assertGreater(ranking[1]["rmse_Nm"], 0.1)

    def test_channel_signature_supports_canonical_order(self):
        audit = _channel_order_audit(load_replay_data())
        self.assertEqual(tuple(audit["inferred_order"]), CANONICAL_JOINT_NAMES)
        self.assertEqual(
            [row["inferred_front_or_hind"] for row in audit["block_command_center_audit"]],
            ["front", "front", "hind", "hind"],
        )
        alignment = audit["internal_field_alignment"]
        self.assertLess(
            alignment["sim_method_vs_real_same_index_rmse_rad"],
            alignment["sim_method_middle_leg_swap_vs_real_rmse_rad"],
        )

    def test_torque_metrics_reports_joint_values(self):
        logged = np.zeros((3, 2), dtype=np.float64)
        recomputed = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        result = _torque_metrics(recomputed, logged, ("A", "B"))
        self.assertAlmostEqual(result["overall_rmse_Nm"], np.sqrt(0.5))
        self.assertEqual(result["per_joint"][0]["rmse_Nm"], 1.0)
        self.assertEqual(result["per_joint"][1]["rmse_Nm"], 0.0)


if __name__ == "__main__":
    unittest.main()
