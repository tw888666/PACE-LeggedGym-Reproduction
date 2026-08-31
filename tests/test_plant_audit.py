import unittest

import numpy as np

from pace_stage0.constants import ANYMAL_ASSET_FILE, ANYMAL_ASSET_ROOT
from pace_stage0.plant_audit import (
    _aggregate_urdf_inertials,
    _collapse_mapping,
    _comparison,
    _parse_urdf,
)


class PlantAuditPureUtilitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = _parse_urdf(ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE)

    def test_public_urdf_inventory(self):
        self.assertEqual(self.urdf["link_count"], 87)
        self.assertEqual(self.urdf["joint_count"], 86)
        self.assertEqual(self.urdf["movable_joint_count"], 12)
        self.assertEqual(self.urdf["fixed_joint_count"], 74)

    def test_collapsed_mapping_conserves_mass(self):
        runtime = ["base"]
        for leg in ("LF", "LH", "RF", "RH"):
            runtime.extend((f"{leg}_HIP", f"{leg}_THIGH", f"{leg}_SHANK"))
        mapping = _collapse_mapping(self.urdf, runtime)
        aggregate = _aggregate_urdf_inertials(self.urdf, mapping)
        self.assertEqual(set(aggregate), set(runtime))
        self.assertAlmostEqual(
            sum(body["mass"] for body in aggregate.values()),
            self.urdf["total_urdf_mass"],
            places=10,
        )
        self.assertEqual(mapping["RF_HIP"]["runtime_body"], "RF_HIP")
        self.assertEqual(mapping["RF_HFE_output"]["runtime_body"], "RF_HIP")
        self.assertEqual(mapping["RF_HFE_drive"]["runtime_body"], "RF_THIGH")
        self.assertEqual(mapping["RF_THIGH"]["runtime_body"], "RF_THIGH")

    def test_no_collapse_maps_each_link_to_itself(self):
        runtime = list(self.urdf["links"])
        mapping = _collapse_mapping(self.urdf, runtime)
        self.assertTrue(all(name == item["runtime_body"] for name, item in mapping.items()))

    def test_ab_comparison_does_not_select_baseline(self):
        def case(value):
            block = {
                "overall_rmse": value,
                "per_joint_rmse": {},
                "focus_joint_rmse": {},
            }
            return {
                "teacher_public": {
                    "windows": {
                        "full_trajectory": {
                            "position_rad": block,
                            "velocity_rad_s": block,
                            "acceleration_rad_s2": block,
                        }
                    }
                }
            }

        result = _comparison(case(1.0), case(0.9))
        self.assertTrue(result["material_position_improvement"])
        self.assertAlmostEqual(result["position_rad"]["percent_change"], -10.0)


if __name__ == "__main__":
    unittest.main()
