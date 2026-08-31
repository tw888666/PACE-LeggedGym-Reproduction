import unittest

from pace_stage0.constants import FIT_PATH
from pace_stage0.fit_order_diagnostics import (
    BLOCK_SLICES,
    PublishedValue,
    _cross_file_audit,
    audit_block,
    build_fit_order_report,
    inspect_fitting_payload,
)


class FitOrderDiagnosticsTest(unittest.TestCase):
    def test_rounding_half_width_respects_display_and_scale(self):
        self.assertAlmostEqual(PublishedValue("0.123").half_rounding_unit, 0.0005)
        self.assertAlmostEqual(
            PublishedValue("76", "0.001").half_rounding_unit, 0.0005
        )
        self.assertAlmostEqual(PublishedValue("-0.0072").half_rounding_unit, 0.00005)

    def test_payload_schema_is_complete_and_has_no_joint_metadata(self):
        schema, params = inspect_fitting_payload(FIT_PATH)
        self.assertEqual(
            schema["top_level_keys"],
            ["params", "params_buffer", "scores", "worst_scores", "bounds"],
        )
        self.assertFalse(schema["machine_readable_joint_metadata"])
        self.assertEqual(params.shape, (49,))
        children = schema["payload"]["children"]
        self.assertEqual(children["params"]["shape"], [49])
        self.assertEqual(children["params_buffer"]["shape"], [100, 49])
        self.assertEqual(children["scores"]["shape"], [100])
        self.assertEqual(children["worst_scores"]["shape"], [100])
        self.assertEqual(children["bounds"]["shape"], [49, 2])

    def test_hypothesis_a_wins_all_table6_blocks(self):
        _, params = inspect_fitting_payload(FIT_PATH)
        for block_name, block_slice in BLOCK_SLICES.items():
            with self.subTest(block=block_name):
                result = audit_block(block_name, params[block_slice])
                self.assertEqual(
                    result["comparison"]["winner_by_rmse"],
                    "H_A_LF_LH_RF_RH",
                )
                a = result["statistics"]["H_A_LF_LH_RF_RH"]
                b = result["statistics"]["H_B_LF_RF_LH_RH"]
                self.assertLess(a["rmse_discrepancy_SI"], b["rmse_discrepancy_SI"])

    def test_report_rejects_direct_order_replay_condition(self):
        report = build_fit_order_report()
        self.assertEqual(report["conclusion"]["most_likely_file_order"], "LF LH RF RH")
        self.assertTrue(report["conclusion"]["RAW_TO_CANONICAL_necessary"])
        self.assertFalse(report["conclusion"]["H_B_strongly_supported"])
        self.assertFalse(report["conditional_direct_order_replay"]["executed"])
        self.assertEqual(report["stage_status"]["Stage_0C"], "FAIL")
        self.assertEqual(
            report["rf_lh_focus_slots"]["aggregate_lower_discrepancy_counts"],
            {"H_A": 21, "H_B": 3, "tie": 0, "comparison_count": 24},
        )

    def test_authorized_cross_file_search_has_only_frozen_fit(self):
        audit = _cross_file_audit()
        self.assertEqual(audit["authorized_glob"], "**/anymal*/fitting.npy")
        self.assertEqual(audit["match_count"], 1)
        self.assertEqual(audit["matches"][0]["params_shape"], [49])
        self.assertTrue(audit["matches"][0]["same_49_parameter_schema"])


if __name__ == "__main__":
    unittest.main()
