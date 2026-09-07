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


if __name__ == "__main__":
    unittest.main()
