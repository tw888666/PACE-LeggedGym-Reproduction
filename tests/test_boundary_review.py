import unittest

from pace_stage0.boundary_review import (
    EXACT_OVERALL_THRESHOLD,
    EXACT_PER_JOINT_THRESHOLD,
    _options,
    _provenance_matrix,
    _readiness_checklist,
)


class BoundaryReviewProtocolTest(unittest.TestCase):
    def test_exact_thresholds_are_unchanged(self):
        self.assertEqual(EXACT_OVERALL_THRESHOLD, 0.010)
        self.assertEqual(EXACT_PER_JOINT_THRESHOLD, 0.020)

    def test_option_c_is_recommended_and_b_is_rejected(self):
        options = {item["option"]: item for item in _options()}
        self.assertEqual(options["B"]["assessment"], "REJECT")
        self.assertEqual(options["C"]["assessment"], "RECOMMENDED")
        self.assertIn("Preserve Stage 0C-E FAIL", options["C"]["policy"])

    def test_readiness_is_evidence_checklist_not_new_rmse_gate(self):
        exact = {
            "one_step_position_rmse_rad": 2.5e-5,
            "one_step_velocity_rmse_rad_s": 0.011,
            "one_step_acceleration_rmse_rad_s2": 4.4,
            "overall_q_rmse_rad": 0.0128,
            "ours_vs_real_rmse_rad": 0.0236,
            "author_vs_real_rmse_rad": 0.0158,
            "sim_nothing_vs_real_rmse_rad": 0.202,
        }
        deterministic = {
            "selected_metric_count": 32,
            "maximum_absolute_metric_difference": 0.0,
        }
        checklist = {item["id"]: item for item in _readiness_checklist(exact, deterministic)}
        self.assertEqual(
            checklist["R4"]["assessment"],
            "SATISFIED_AS_EVIDENCE_NOT_A_NEW_THRESHOLD",
        )
        self.assertIn("FAIL", checklist["R5"]["evidence"])
        self.assertEqual(
            checklist["R10"]["assessment"],
            "SATISFIABLE_ONLY_IF_MANDATORY_BANNER_IS_ADOPTED",
        )

    def test_provenance_matrix_contains_all_three_tiers(self):
        tiers = {item["tier"] for item in _provenance_matrix()}
        self.assertEqual(
            tiers,
            {
                "A_author_confirmed_or_released_source",
                "B_strong_inference",
                "C_unresolved",
            },
        )


if __name__ == "__main__":
    unittest.main()
