import json
import tempfile
import unittest
from pathlib import Path

from pace_stage1.pace_v2_profiles import (
    ProtocolIdentity,
    leg_gym_reference_profile,
    pace_v2_formal_profile,
    pace_v2_validation_profile,
)
from pace_stage1.protocol_gate import (
    DEFAULT_PROTOCOL_MATRIX_PATH,
    formal_training_status,
    require_formal_training_allowed,
    require_validation_training_allowed,
)


class PaceV2ProfileTest(unittest.TestCase):
    def test_legacy_reference_values_remain_separate(self):
        profile = leg_gym_reference_profile()
        self.assertEqual(profile.identity, ProtocolIdentity.LEG_GYM_REFERENCE)
        self.assertEqual(profile.ppo.actor_hidden_dims, (512, 256, 128))
        self.assertEqual(profile.ppo.num_mini_batches, 4)
        self.assertEqual(profile.ppo.entropy.initial, 0.01)
        self.assertFalse(profile.ppo.empirical_normalization)
        self.assertTrue(profile.allows_short_validation)
        self.assertFalse(profile.allows_formal_training)

    def test_validation_uses_table8_without_making_30k_the_default(self):
        profile = pace_v2_validation_profile(max_iterations=250)
        self.assertEqual(profile.identity, ProtocolIdentity.PACE_V2_VALIDATION)
        self.assertEqual(profile.requested_iterations, 250)
        self.assertEqual(profile.ppo.actor_hidden_dims, (256, 256, 256, 128))
        self.assertEqual(profile.ppo.critic_hidden_dims, (256, 256, 256, 128))
        self.assertEqual(profile.ppo.num_mini_batches, 10)
        self.assertEqual(profile.ppo.init_noise_std, 1.5)
        self.assertEqual(profile.ppo.entropy.initial, 0.002)
        self.assertEqual(profile.ppo.entropy.final, 0.0005)
        self.assertEqual(profile.ppo.entropy.turnover_iteration, 20_000)
        self.assertGreater(profile.ppo.entropy.slope_eta, 0)
        self.assertTrue(profile.allows_short_validation)
        self.assertFalse(profile.allows_formal_training)
        self.assertEqual(
            set(profile.unresolved_parameters),
            set(formal_training_status().blocking_items),
        )

    def test_current_matrix_allows_formal_profile_after_approval(self):
        status = formal_training_status()
        self.assertTrue(status.allowed)
        self.assertTrue(status.validation_allowed)
        self.assertEqual(len(status.semantic_blockers), 0)
        self.assertEqual(status.reproducibility_blockers, ())
        self.assertEqual(status.formal_reporting_blockers, ())
        self.assertEqual(status.blocking_items, ())
        self.assertTrue(pace_v2_formal_profile().allows_formal_training)
        self.assertTrue(require_validation_training_allowed().validation_allowed)

    def test_validation_can_open_after_only_semantic_blockers_are_resolved(self):
        payload = json.loads(DEFAULT_PROTOCOL_MATRIX_PATH.read_text(encoding="utf-8"))
        remaining_formal_blockers = []
        for entry in payload["matrix"]:
            if entry.get("blocker_class") == "SEMANTIC":
                entry.pop("blocks_validation_training", None)
                entry.pop("blocks_formal_training", None)
            elif entry.get("blocks_formal_training") is True:
                remaining_formal_blockers.append(entry["id"])
        payload["formal_freeze_blockers"] = remaining_formal_blockers
        payload["baseline"]["validation_training_allowed"] = True
        payload["baseline"]["formal_training_allowed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            status = require_validation_training_allowed(path)
            self.assertTrue(status.validation_allowed)
            self.assertFalse(status.allowed)
            self.assertEqual(status.semantic_blockers, ())
            self.assertEqual(
                status.reproducibility_blockers, ()
            )
            self.assertTrue(
                pace_v2_validation_profile(250, path).allows_short_validation
            )

    def test_gate_fails_closed_when_matrix_approval_is_not_true(self):
        payload = json.loads(DEFAULT_PROTOCOL_MATRIX_PATH.read_text(encoding="utf-8"))
        for entry in payload["matrix"]:
            entry.pop("blocks_formal_training", None)
        payload["formal_freeze_blockers"] = []
        payload["baseline"]["formal_training_allowed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                require_formal_training_allowed(path)

    def test_formal_profile_requires_separate_explicit_approval(self):
        payload = json.loads(DEFAULT_PROTOCOL_MATRIX_PATH.read_text(encoding="utf-8"))
        for entry in payload["matrix"]:
            entry.pop("blocks_formal_training", None)
        payload["formal_freeze_blockers"] = []
        payload["baseline"]["formal_training_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(pace_v2_formal_profile(path).allows_formal_training)


if __name__ == "__main__":
    unittest.main()
