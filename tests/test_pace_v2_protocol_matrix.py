import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "provenance" / "gpt-pace-v2-protocol-matrix.json"


class PaceV2ProtocolMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    def test_schema_and_training_gate_are_explicit(self):
        self.assertEqual(
            self.matrix["schema"], "pace_v2_protocol_provenance_matrix.v1"
        )
        self.assertFalse(self.matrix["baseline"]["validation_training_allowed"])
        self.assertFalse(self.matrix["baseline"]["formal_training_allowed"])

    def test_matrix_ids_and_classifications_are_valid(self):
        entries = self.matrix["matrix"]
        entry_ids = [entry["id"] for entry in entries]
        self.assertEqual(len(entry_ids), len(set(entry_ids)))

        provenance_classes = set(self.matrix["provenance_classes"])
        implementation_states = set(self.matrix["implementation_states"])
        for entry in entries:
            self.assertIn(entry["provenance"], provenance_classes)
            self.assertIn(entry["implementation_state"], implementation_states)

    def test_all_freeze_blockers_reference_matrix_entries(self):
        entry_ids = {entry["id"] for entry in self.matrix["matrix"]}
        blockers = set(self.matrix["formal_freeze_blockers"])
        immediate_scope = set(
            self.matrix["paper_exact_infrastructure_ready"]
        )
        self.assertTrue(blockers)
        self.assertLessEqual(blockers, entry_ids)
        self.assertLessEqual(immediate_scope, entry_ids)

    def test_required_provenance_classes_are_represented(self):
        represented = {entry["provenance"] for entry in self.matrix["matrix"]}
        self.assertEqual(represented, set(self.matrix["provenance_classes"]))

    def test_formal_blockers_have_impact_class_and_validation_semantics(self):
        blockers = [
            entry
            for entry in self.matrix["matrix"]
            if entry.get("blocks_formal_training") is True
        ]
        classes = {name: [] for name in self.matrix["blocker_classes"]}
        for entry in blockers:
            classes[entry["blocker_class"]].append(entry["id"])
            self.assertIn(entry["resolution_priority"], (1, 2, 3, 4))
            self.assertIn(
                entry["resolution_policy"], self.matrix["resolution_policies"]
            )
            self.assertEqual(
                entry["blocks_validation_training"],
                entry["blocker_class"] == "SEMANTIC",
            )

        self.assertEqual(len(classes["SEMANTIC"]), 11)
        self.assertEqual(classes["REPRODUCIBILITY"], ["ppo.entropy_slope_eta"])
        self.assertEqual(classes["FORMAL_REPORTING"], [])

    def test_action_physical_semantics_are_separate_from_network_mapping(self):
        entries = {entry["id"]: entry for entry in self.matrix["matrix"]}
        for entry_id in (
            "control.action_physical_semantics",
            "control.target_mapping",
        ):
            self.assertEqual(entries[entry_id]["provenance"], "PAPER_EXACT")
            self.assertEqual(entries[entry_id]["implementation_state"], "MATCH")
            self.assertNotIn("blocks_formal_training", entries[entry_id])

        for entry_id in (
            "control.network_output_to_action_offset_mapping",
            "control.action_output_clipping",
            "control.default_posture_q0",
        ):
            self.assertEqual(entries[entry_id]["provenance"], "UNRESOLVED")
            self.assertTrue(entries[entry_id]["blocks_validation_training"])
            self.assertTrue(entries[entry_id]["blocks_formal_training"])

    def test_public_sysid_action_config_is_not_locomotion_evidence(self):
        entry = next(
            item
            for item in self.matrix["matrix"]
            if item["id"] == "control.public_pace_sim2real_action_cfg_scope"
        )
        self.assertEqual(entry["provenance"], "PUBLIC_CODE_CONTEXT_ONLY")
        self.assertFalse(entry["evidence_scope"]["locomotion_evidence"])
        self.assertNotIn("blocks_formal_training", entry)

    def test_action_evidence_audit_preserves_unresolved_status_and_gates(self):
        audit = self.matrix["action_evidence_audit"]
        self.assertEqual(
            audit["status"],
            "SEARCHED_NO_AUTHORITATIVE_LOCOMOTION_VALUE_FOUND",
        )
        self.assertEqual(audit["gate_effect"], "NONE")
        self.assertFalse(self.matrix["baseline"]["validation_training_allowed"])
        self.assertFalse(self.matrix["baseline"]["formal_training_allowed"])

        entries = {entry["id"]: entry for entry in self.matrix["matrix"]}
        self.assertEqual(set(audit["still_unresolved"]), {
            "control.network_output_to_action_offset_mapping",
            "control.action_output_clipping",
            "control.default_posture_q0",
        })
        for entry_id in audit["still_unresolved"]:
            self.assertEqual(entries[entry_id]["provenance"], "UNRESOLVED")
            self.assertEqual(
                entries[entry_id]["evidence_audit_status"],
                audit["status"],
            )
            self.assertIn(entry_id, self.matrix["formal_freeze_blockers"])

        self.assertEqual(len(self.matrix["formal_freeze_blockers"]), 12)


if __name__ == "__main__":
    unittest.main()
