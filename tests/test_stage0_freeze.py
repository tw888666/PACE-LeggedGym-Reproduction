import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "provenance" / "stage0_freeze.json"
README_PATH = REPO_ROOT / "README.md"


class Stage0FreezeManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_statuses_preserve_exact_failure_and_approved_readiness(self):
        status = self.manifest["status"]
        self.assertEqual(status["Stage_0C_E_exact_legacy_replay"], "FAIL / unresolved")
        self.assertEqual(status["Stage_0C_R_reproduction_readiness"], "PASS")
        self.assertEqual(status["Stage_0"], "FROZEN")
        self.assertEqual(self.manifest["stage0C_R"]["approval_type"], "explicit_human_approval")
        self.assertEqual(self.manifest["stage0C_R"]["approved_option"], "C")

    def test_exact_thresholds_and_residual_are_unchanged(self):
        exact = self.manifest["exact_replay"]
        gates = exact["gates"]
        self.assertEqual(gates["overall_encoder_RMSE_le_0p010"]["threshold_rad"], 0.010)
        self.assertEqual(gates["all_per_joint_RMSE_le_0p020"]["threshold_rad"], 0.020)
        self.assertFalse(exact["thresholds_relaxed"])
        self.assertEqual(
            exact["metrics"]["overall_encoder_frame_RMSE_rad"],
            0.012818364796375577,
        )

    def test_stage1_is_authorized_without_starting_ppo(self):
        authorization = self.manifest["Stage_1_authorization"]
        self.assertTrue(authorization["locomotion_implementation_authorized"])
        self.assertFalse(authorization["PPO_training_started"])
        self.assertFalse(self.manifest["status"]["PPO_training_started"])
        self.assertIn("PPO implementation", authorization["not_performed_in_freeze_commit"])

    def test_required_artifact_hashes_are_present_and_match(self):
        artifacts = self.manifest["artifact_identity"]
        expected = [
            (
                Path(artifacts["fitting_npy"]["path"]),
                "4436941fa5e9a5e8e1ef93d55956fcffdb4c4c4526b8ef3e145e6ea619fbe1c8",
            ),
            (
                Path(artifacts["data_npy"]["path"]),
                "edf2e7648602802c87bbe52bb5c098c41553e8c683b2d9013d6f3f4df8bca621",
            ),
            (
                REPO_ROOT / artifacts["ANYmal_public_asset"]["vendored_URDF"],
                "0f7c208ab53d595a70711034b9ebe4dd47a4536ed0ccb1cc08d732c3ecd7ce1a",
            ),
        ]
        for path, expected_sha in expected:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_sha)

    def test_unresolved_provenance_is_explicit_and_uniform(self):
        unresolved = self.manifest["unresolved_provenance"]
        self.assertEqual(len(unresolved), 6)
        self.assertEqual(
            {item["status"] for item in unresolved},
            {"UNRESOLVED / NOT PUBLICLY RECOVERED"},
        )
        items = {item["item"] for item in unresolved}
        self.assertIn("legacy ANYmal exact asset identity/hash", items)
        self.assertIn("paper-era complete PhysX configuration", items)

    def test_all_audited_hypotheses_remain_closed_without_formal_change(self):
        hypotheses = self.manifest["closed_hypotheses"]
        self.assertEqual(len(hypotheses), 20)
        self.assertTrue(all(item["formal_change"] is False for item in hypotheses))

    def test_readiness_checklist_is_complete(self):
        checklist = self.manifest["stage0C_R"]["checklist"]
        self.assertEqual([item["id"] for item in checklist], [f"R{i}" for i in range(1, 11)])
        self.assertEqual({item["status"] for item in checklist}, {"PASS"})

    def test_freeze_commit_uses_explicit_non_self_referential_resolution(self):
        freeze_commit = self.manifest["freeze_identity"]["freeze_commit"]
        self.assertEqual(freeze_commit["resolution"], "annotated_tag_target")
        self.assertEqual(freeze_commit["annotated_tag"], "stage0-public-reproduction-ready")
        self.assertIsNone(freeze_commit["inline_sha"])

    def test_readme_distinguishes_exact_and_readiness_tracks(self):
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("Stage 0C-E     FAIL / unresolved", readme)
        self.assertIn("Stage 0C-R     PASS", readme)
        self.assertIn("exact legacy-trajectory replay gate", readme)
        self.assertIn("public-information reproduction-readiness", readme)
        self.assertIn("PPO training   NOT STARTED", readme)


if __name__ == "__main__":
    unittest.main()
