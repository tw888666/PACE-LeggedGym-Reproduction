import unittest
from types import SimpleNamespace

import numpy as np

from pace_stage0.bias_law_counterfactual import (
    _classify,
    _torque_metrics,
    _window_metrics,
)


class BiasLawCounterfactualMetricsTest(unittest.TestCase):
    def test_torque_alignment_excludes_fifo_initialization(self):
        applied = np.asarray(
            [[100.0], [101.0], [102.0], [3.0], [4.0]], dtype=np.float64
        )
        logged = np.asarray(
            [[-1.0], [-1.0], [-1.0], [-1.0], [3.0], [4.0]], dtype=np.float64
        )
        case = {"arrays": {"applied_torque": applied}}
        data = SimpleNamespace(sim_method_dof_torques=logged)
        metrics = _torque_metrics(case, data, delay_steps=3)
        self.assertEqual(metrics["active_candidate_interval_start"], 3)
        self.assertEqual(metrics["active_dataset_state_start"], 4)
        self.assertEqual(metrics["active_sample_count"], 2)
        self.assertEqual(metrics["overall_rmse_Nm"], 0.0)

    def test_window_metrics_uses_requested_prefixes_and_full(self):
        ours = np.arange(24, dtype=np.float64).reshape(12, 2)
        result = _window_metrics(ours, ours + 1.0, windows=(3, 5))
        self.assertEqual(result["first_3"]["sample_count"], 3)
        self.assertEqual(result["first_5"]["sample_count"], 5)
        self.assertEqual(result["full"]["sample_count"], 12)
        self.assertEqual(result["full"]["overall_rmse"], 1.0)

    def test_case_c_when_legacy_position_materially_worsens(self):
        def mode(position, torque, correlation, gate=False):
            return {
                "position": {"full": {"overall_rmse": position}},
                "torque": {
                    "overall_rmse_Nm": torque,
                    "overall_correlation": correlation,
                },
                "gates": {
                    "overall_position_le_0p010": gate,
                    "all_per_joint_position_le_0p020": gate,
                },
            }

        decision = _classify(
            mode(0.0128, 0.86, 0.998),
            mode(0.0151, 0.85, 0.998),
        )
        self.assertEqual(decision["case"], "C")
        self.assertTrue(decision["branch_A_closed"])
        self.assertTrue(decision["asset_plant_audit_authorized"])
        self.assertFalse(decision["formal_actuator_change_applied"])


if __name__ == "__main__":
    unittest.main()
