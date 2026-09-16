import math
import tempfile
import unittest
from pathlib import Path
import torch
from test_normalization_bridge import ComponentRunner, TensorEnv
from pace_stage1.pace_v2_runner import ValidationLoopMixin, validation_train_cfg
from pace_stage1.pace_v2_entropy import pace_v2_entropy_coefficient
from pace_stage1.pace_v2_profiles import PaceV2EntropyConfig

class ClockEnv(TensorEnv):
    def set_training_iteration(self,i): self.training_iteration=i
class ClockRunner(ValidationLoopMixin,ComponentRunner): pass

class EntryTest(unittest.TestCase):
    def test_primary_schedule_geometry(self):
        c=PaceV2EntropyConfig()
        values=[pace_v2_entropy_coefficient(i,c) for i in range(30001)]
        self.assertAlmostEqual(c.slope_eta, math.atanh(.8)/10000,places=16)
        self.assertAlmostEqual(values[0],.0019817073170731707,places=16)
        self.assertEqual(values[20000],.00125)
        self.assertAlmostEqual(values[30000],.00065,places=16)
        self.assertTrue(all(.0005<=v<=.002 for v in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
    def test_resume_uses_next_update_not_last_update(self):
        torch.set_num_threads(1)
        cfg=validation_train_cfg(10)
        cfg['policy'].update(actor_hidden_dims=[8],critic_hidden_dims=[8])
        cfg['runner']['num_steps_per_env']=2
        cfg['algorithm'].update(num_mini_batches=2,num_learning_epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            first=ClockRunner(ClockEnv(),cfg,log_dir=tmp)
            first.current_learning_iteration=12344
            first.iteration()
            file=Path(tmp)/'clock.pt'
            first.save(file)
            restored=ClockRunner(ClockEnv(),cfg,log_dir=tmp)
            restored.load(file)
            a=first.iteration();b=restored.iteration()
            self.assertEqual(a['iteration'],12345)
            self.assertEqual(a['entropy_coef'],b['entropy_coef'])
            self.assertEqual(b['entropy_coef'],pace_v2_entropy_coefficient(12345,PaceV2EntropyConfig()))
            self.assertEqual(restored.env.training_iteration,12345)
            payload=torch.load(file);payload['iter']+=1;torch.save(payload,file)
            with self.assertRaisesRegex(RuntimeError,'iteration mismatch'): restored.load(file)
