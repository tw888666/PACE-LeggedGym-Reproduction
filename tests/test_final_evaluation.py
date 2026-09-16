"""Final-evaluation component checks; run in a fresh process for Isaac Gym import order."""
from pace_stage1.final_evaluation import evaluate_group
from pace_stage1.final_evaluation_env import case_random, FinalEvaluationEnv
from pace_stage1.final_evaluation_report import tails
from pace_stage1.semantics import build_actor_observation, quat_rotate_inverse
from pace_stage1.config import STAGE1_CONFIG
from pace_stage1.final_evaluation_batch import run as run_batch
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import json


class FinalEvaluationChecks(unittest.TestCase):
    def test_case_streams_are_independent_of_batch_order(self):
        a={'seed':'200000','case_id':'123'}
        b={'seed':'200001','case_id':'456'}
        forward=[case_random(c,1,(5,48)) for c in (a,b)]
        backward=[case_random(c,1,(5,48)) for c in (b,a)]
        np.testing.assert_array_equal(forward[0],backward[1])
        np.testing.assert_array_equal(forward[1],backward[0])
        np.testing.assert_array_equal(forward[0][:2],case_random(a,1,(2,48)))
        self.assertFalse(np.array_equal(forward[0],case_random(a,2,(5,48))))

    def test_actor_observation_matches_frozen_builder_with_zero_noise(self):
        n=2
        root=torch.zeros(n,13);root[:,6]=1;root[:,7:13]=torch.arange(12).reshape(2,6)/10
        fake=SimpleNamespace(root_states=root,num_envs=n,device='cpu',
            commands=torch.tensor([[.25,0,0,0],[0,.75,.5,0]]),
            dof_pos=torch.ones(n,12)*.4,dof_vel=torch.ones(n,12)*2,
            actions=torch.ones(n,12)*.3,actuator=SimpleNamespace(encoder_bias=torch.ones(12)*.02),
            cfg=STAGE1_CONFIG,noise=torch.full((1,n,48),.5),common_step_counter=0)
        FinalEvaluationEnv._observations(fake)
        gravity=torch.tensor([[0.,0.,-1.]]).repeat(n,1)
        expected=build_actor_observation(root[:,7:10],root[:,10:13],gravity,fake.commands,
            fake.dof_pos-fake.actuator.encoder_bias,fake.dof_vel,fake.actions,
            add_noise=False,config=STAGE1_CONFIG)
        self.assertTrue(torch.equal(fake.obs_buf,expected))

    def test_busy_group_does_not_execute_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with (directory/'gpt-执行锁').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('pace_stage1.final_evaluation._evaluate_group') as inner:
                    self.assertFalse(evaluate_group([],directory,Path('unused.pt')))
                    inner.assert_not_called()
            with patch('pace_stage1.final_evaluation._evaluate_group') as inner:
                self.assertTrue(evaluate_group([],directory,Path('unused.pt')))
                inner.assert_called_once()

    def test_tail_thresholds_are_strict(self):
        result=tails(np.array([0.,2.,5.,10.,11.]))
        self.assertEqual(result['scalar_count'],5)
        self.assertEqual(result['prob_gt2'],.6)
        self.assertEqual(result['prob_gt5'],.4)
        self.assertEqual(result['prob_gt10'],.2)

    def test_homogeneous_tiles_have_separate_origins(self):
        env=FinalEvaluationEnv.__new__(FinalEvaluationEnv)
        env.cases=[dict(seed='200000',case_id=str(i),source_column='2',difficulty_row='5') for i in range(64)]
        env.cache='unused';env.terrain_mode='trimesh';env.sim=None
        env.gym=SimpleNamespace(add_triangle_mesh=lambda *args:None)
        tile=np.arange(80*80,dtype=np.int16).reshape(80,80)
        with patch('pace_stage1.final_evaluation_env.source_tile',return_value=tile),patch('pace_stage1.final_evaluation_env.terrain_utils.convert_heightfield_to_trimesh',return_value=(np.zeros((3,3)),np.zeros((1,3)))):
            env._create_terrain()
        origins=env.case_origins[:,:2]
        distances=np.linalg.norm(origins[:,None]-origins[None,:],axis=-1)
        self.assertEqual(distances[distances>0].min(),8.)
        self.assertEqual(len(np.unique(origins,axis=0)),64)
        np.testing.assert_array_equal((origins-origins[0])%8,0)

    def test_native_missing_interactions_invalidates_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            directory=output/'gpt-00-smooth_slope-0'
            def fake_process(command,stdout,stderr):
                (directory/'gpt-完成记录.json').write_text(json.dumps({'state':'completed'}))
                stdout.write('PhysX: simulation will miss interactions\n')
                return SimpleNamespace(returncode=0)
            with patch('pace_stage1.final_evaluation_batch.subprocess.run',side_effect=fake_process):
                with self.assertRaisesRegex(RuntimeError,'runtime audit'):
                    run_batch(output,group=0)
            self.assertFalse((directory/'gpt-完成记录.json').exists())
            self.assertTrue((directory/'gpt-无效完成记录.json').exists())


if __name__=='__main__':
    unittest.main()
