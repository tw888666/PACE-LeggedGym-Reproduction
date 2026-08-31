import unittest

import numpy as np

from pace_stage0.constants import (
    EXPECTED_DATA_SHA256,
    EXPECTED_FIT_SHA256,
    RAW_JOINT_NAMES,
    RAW_TO_CANONICAL,
    canonical_gather_indices,
)
from pace_stage0.data import load_replay_data
from pace_stage0.decoder import decode_fit
from pace_stage0.replay import _frame_sanity


class TestDecoder(unittest.TestCase):
    def test_frozen_hash_layout_and_delay(self):
        decoded = decode_fit()
        self.assertEqual(decoded.sha256, EXPECTED_FIT_SHA256)
        self.assertEqual(decoded.params_raw.shape, (49,))
        self.assertEqual(decoded.bounds_raw.shape, (49, 2))
        self.assertAlmostEqual(decoded.delay_raw, 3.2405974755, places=9)
        self.assertEqual(decoded.delay_steps, 3)

    def test_each_legacy_block_is_reordered_independently(self):
        decoded = decode_fit()
        indices = np.asarray(RAW_TO_CANONICAL)
        np.testing.assert_array_equal(decoded.friction, decoded.params_raw[0:12][indices])
        np.testing.assert_array_equal(decoded.damping, decoded.params_raw[12:24][indices])
        np.testing.assert_array_equal(decoded.armature, decoded.params_raw[24:36][indices])
        np.testing.assert_array_equal(decoded.encoder_bias, decoded.params_raw[36:48][indices])

    def test_replay_data_is_already_canonical_and_aligned(self):
        data = load_replay_data()
        self.assertEqual(data.sha256, EXPECTED_DATA_SHA256)
        self.assertEqual(data.real_dof_pos.shape, (6680, 12))
        self.assertEqual(data.real_des_dof_pos.shape, (6680, 12))
        self.assertEqual(data.sim_method_dof_pos.shape, data.real_dof_pos.shape)
        self.assertEqual(data.sim_nothing_dof_pos.shape, data.real_dof_pos.shape)

    def test_asset_raw_order_gathers_to_canonical(self):
        self.assertEqual(canonical_gather_indices(RAW_JOINT_NAMES), RAW_TO_CANONICAL)

    def test_legacy_frame_sanity_is_reported_without_auto_selection(self):
        sanity = _frame_sanity(load_replay_data(), decode_fit())
        self.assertFalse(sanity["encoder_frame_lower_than_raw"])
        self.assertGreater(
            sanity["rmse_sim_method_minus_bias_vs_real"],
            sanity["rmse_sim_method_true_vs_real"],
        )


if __name__ == "__main__":
    unittest.main()
