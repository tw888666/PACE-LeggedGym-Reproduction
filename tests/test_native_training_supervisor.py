import sys
import tempfile
from pathlib import Path
import unittest
from pace_stage1.native_training_supervisor import supervise


class NativeSupervisorTest(unittest.TestCase):
    def test_native_c_warning_is_seen_before_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source="import ctypes,time; ctypes.CDLL(None).printf(b'PhysX simulation will miss interactions\\n'); time.sleep(30)"
            result=supervise(['stdbuf','-oL','-eL',sys.executable,'-u','-c',source],root/'gpt-native.log',root/'gpt-native.json')
            self.assertEqual(result['state'],'FAILED')
            self.assertLess(result['seconds'],10.)

    def test_native_warning_cannot_pass_even_if_child_would_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=supervise([sys.executable,'-u','-c',"print('PhysX: simulation will miss interactions',flush=True)"],root/'gpt-log.log',root/'gpt-status.json')
            self.assertEqual(result['state'],'FAILED')
            self.assertIsNotNone(result['native_error'])

    def test_clean_child_and_nonzero_exit_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            clean=supervise([sys.executable,'-c',"print('finished')"],root/'gpt-clean.log',root/'gpt-clean.json')
            failed=supervise([sys.executable,'-c','raise SystemExit(2)'],root/'gpt-failed.log',root/'gpt-failed.json')
            self.assertEqual(clean['state'],'PROCESS_COMPLETED')
            self.assertEqual(failed['state'],'FAILED')
            self.assertEqual(failed['returncode'],2)


if __name__=='__main__':unittest.main()
