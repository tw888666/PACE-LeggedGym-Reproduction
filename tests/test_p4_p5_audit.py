import unittest

import numpy as np

from pace_stage0.p4_p5_audit import (
    IntegrationConfig,
    _joint_axis_signs,
    _regression_structure,
    _scenarios,
    _sensitivity,
    _timestamp_jitter_diagnostic,
    _variants,
)


def _evaluation(qdot=1.0, acceleration=1.0, h32=1.0, h128=1.0, full=1.0):
    return {
        "one_step": {
            "position_rmse_rad": 1.0,
            "velocity_rmse_rad_s": qdot,
            "acceleration_rmse_rad_s2": acceleration,
        },
        "horizons": {
            "H32": {"position_rmse_rad": h32},
            "H128": {"position_rmse_rad": h128},
            "full": {"position_rmse_rad": full},
        },
    }


class P4P5AuditUtilitiesTest(unittest.TestCase):
    def test_frozen_variant_and_scenario_grid(self):
        self.assertEqual(
            [variant.name for variant in _variants()],
            ["baseline", "damping0", "friction0", "both0"],
        )
        self.assertEqual(
            [scenario.name for scenario in _scenarios()],
            ["one_step", "H8", "H32", "H128", "full"],
        )
        self.assertEqual(_scenarios()[0].reset_horizon, 1)
        self.assertTrue(_scenarios()[0].author_initial_state)

    def test_material_change_requires_ten_percent(self):
        baseline = _evaluation()
        changed = _sensitivity(_evaluation(qdot=0.89, h32=0.99), baseline)
        self.assertTrue(changed["material_change"])
        self.assertTrue(changed["evidence_backed_material_improvement"])
        wrong_direction = _sensitivity(_evaluation(qdot=0.89, h32=1.01), baseline)
        self.assertTrue(wrong_direction["material_change"])
        self.assertFalse(wrong_direction["evidence_backed_material_improvement"])

    def test_h32_material_improvement_needs_local_direction(self):
        baseline = _evaluation()
        coherent = _sensitivity(_evaluation(qdot=0.99, h32=0.89), baseline)
        self.assertTrue(coherent["evidence_backed_material_improvement"])
        incoherent = _sensitivity(_evaluation(qdot=1.01, acceleration=1.01, h32=0.89), baseline)
        self.assertFalse(incoherent["evidence_backed_material_improvement"])

    def test_regression_reports_sign_and_r_squared_without_slope(self):
        qdot = np.linspace(-1.0, 1.0, 100)
        result = _regression_structure(3.0 * qdot + 0.5, qdot)
        self.assertEqual(result["slope_sign"], "positive")
        self.assertGreater(result["r_squared"], 0.999999)
        self.assertNotIn("slope", result)
        self.assertFalse(result["slope_written_back_to_model"])

    def test_frozen_urdf_mirror_axis_signs(self):
        axes = _joint_axis_signs()
        self.assertEqual(axes["RF_KFE"]["motion_alignment_sign"], -1.0)
        self.assertEqual(axes["LH_KFE"]["motion_alignment_sign"], 1.0)

    def test_timestamp_jitter_is_diagnostic_only(self):
        count = 10
        time = np.arange(count, dtype=np.float64) * 0.0025
        time[1::2] += 1e-5
        residual = np.ones((count - 1, 12), dtype=np.float64)
        evaluation = {
            "arrays": {
                "velocity_residual": residual,
                "acceleration_residual": residual / 0.0025,
            }
        }
        data = type(
            "Data",
            (),
            {
                "time": time,
                "sim_method_dof_pos": np.zeros((count, 12)),
                "sim_method_dof_vel": np.zeros((count, 12)),
            },
        )()
        result = _timestamp_jitter_diagnostic(evaluation, data)
        self.assertFalse(result["variable_dt_simulation_run"])
        self.assertEqual(result["simulation_dt_remains_fixed_s"], 0.0025)

    def test_p5_configs_do_not_change_control_dt(self):
        substeps = IntegrationConfig("substeps2", substeps=2)
        velocity = IntegrationConfig("velocity_iterations1", num_velocity_iterations=1)
        self.assertEqual(substeps.substeps, 2)
        self.assertEqual(substeps.num_velocity_iterations, 0)
        self.assertEqual(velocity.substeps, 1)
        self.assertEqual(velocity.num_velocity_iterations, 1)


if __name__ == "__main__":
    unittest.main()
