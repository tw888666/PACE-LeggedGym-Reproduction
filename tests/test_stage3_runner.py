import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from pace_stage1.pace_v2_runner import validation_train_cfg
from pace_stage1.stage3_lagrangian import ConstraintConfig
from pace_stage1.stage3_runner import Stage3Runner


class ToyCostEnv:
    num_envs, num_obs, num_privileged_obs, num_actions = 4, 2, 3, 1
    device, policy_dt = 'cpu', .01
    cost_spec = SimpleNamespace(definition='pace_corrected')

    def reset(self):
        self.obs = torch.zeros(4, 2)
        self.critic = torch.zeros(4, 3)
        return self.obs, self.critic

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic

    def set_training_iteration(self, iteration):
        self.iteration = iteration

    def step(self, actions):
        self.obs = torch.randn(4, 2)
        self.critic = torch.randn(4, 3)
        return self.obs, self.critic, torch.ones(4)*.01, torch.zeros(4, dtype=torch.bool), dict(
            costs=torch.tensor([-1., 2., 3., 4.]), cost_definition='pace_corrected')


class Stage3RunnerChecks(unittest.TestCase):
    def test_full_runner_reload_after_inference_and_fresh_process_semantics(self):
        torch.set_num_threads(1)
        cfg = validation_train_cfg(3)
        cfg['runner']['num_steps_per_env'] = 3
        cfg['policy'].update(actor_hidden_dims=[8], critic_hidden_dims=[8])
        cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
        constraint = ConstraintConfig(100.)
        runner = Stage3Runner(ToyCostEnv(), cfg, constraint=constraint)
        runner.iteration()
        count = runner.normalizers.actor.count.item()
        self.assertEqual(count, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            runner.save(path)
            runner.iteration()
            runner.load(path)
            self.assertEqual(runner.current_learning_iteration, 1)
            self.assertEqual(runner.normalizers.actor.count.item(), count)
            row = runner.iteration()
            self.assertEqual(row['next_iteration'], 2)
            fresh = Stage3Runner(ToyCostEnv(), cfg, constraint=constraint)
            fresh.load(path)
            self.assertEqual(fresh.current_learning_iteration, 1)
            row = fresh.iteration()
            self.assertEqual(row['actor_count'], count+16)
            self.assertEqual(row['next_iteration'], 2)


if __name__ == '__main__':
    unittest.main()
