"""Synthetic PPO component tests: no simulator, locomotion training or gate bypass."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO

from pace_stage1.config import ppo_train_cfg, STAGE1_CONFIG
from pace_stage1.empirical_normalization import RunningMeanVarianceNormalizer
from pace_stage1.normalization_bridge import NormalizationRunnerMixin, PaceV2ValidationRunner
from pace_stage1.semantics import build_actor_observation, build_critic_observation

ROOT = Path(__file__).resolve().parents[1]


class TensorEnv:
    num_envs = 4
    num_obs = 48
    num_privileged_obs = 353
    num_actions = 12
    device = 'cpu'
    max_episode_length = 100

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.tick = 0
        self._observe()

    def _observe(self):
        # Deliberately different actor/critic scales exercise independent moments.
        self.obs = torch.arange(192, dtype=torch.float32).reshape(4, 48) / 50 + self.tick / 10
        self.critic = torch.arange(1412, dtype=torch.float32).reshape(4, 353) / 80 - self.tick / 5

    def reset(self):
        self.tick = 0
        self._observe()
        return self.obs, self.critic

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic

    def step(self, actions):
        self.tick += 1
        self._observe()
        return self.obs, self.critic, -actions.square().mean(1), torch.zeros(4), {}


class ComponentRunner(NormalizationRunnerMixin, OnPolicyRunner):
    """Exercise the production mixin against tensors and the real v1.0.2 runner."""


def make_runner(directory):
    cfg = ppo_train_cfg()
    cfg['policy'].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
    cfg['runner'].update(num_steps_per_env=4, save_interval=50)
    cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
    return ComponentRunner(TensorEnv(), cfg, log_dir=str(directory), device='cpu')


def outputs(runner, obs, critic):
    runner.alg.actor_critic.eval()
    with torch.no_grad():
        return {
            'actor_obs': runner.normalizers.actor(obs),
            'critic_obs': runner.normalizers.critic(critic),
            'action': runner.alg.actor_critic.act_inference(obs),
            'value': runner.alg.actor_critic.evaluate(critic),
        }


def worker(directory):
    torch.set_num_threads(1)
    folder = Path(directory)
    runner = make_runner(folder)
    runner.load(folder / 'policy.pt')
    sample = torch.load(folder / 'sample.pt')
    torch.save({
        'outputs': outputs(runner, sample['actor'], sample['critic']),
        'optimizer': runner.alg.optimizer.state_dict(),
        'model': runner.alg.actor_critic.state_dict(),
        'iteration': runner.current_learning_iteration,
        'learning_rate': runner.alg.learning_rate,
    }, folder / 'restored.pt')


class NormalizationCompatibilityTest(unittest.TestCase):
    def test_upstream_arithmetic_and_eval_match_exactly(self):
        spec = importlib.util.spec_from_file_location('upstream_norm', ROOT / 'third_party/rsl_rl_normalization_v3_0_1/normalization.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        local = RunningMeanVarianceNormalizer(3)
        reference = upstream.EmpiricalNormalization(3)
        for x in (torch.tensor([[1., 3., 2.], [2., 3., 2.0001]]), torch.tensor([[4., 3., 2.0002]])):
            local.update(x)
            reference.update(x)
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(local.state_dict()[key], value), key)
            self.assertTrue(torch.equal(local(x), reference(x)))
        local.eval()
        saved = copy.deepcopy(local.state_dict())
        local.update(torch.ones(2, 3) * 999)
        result = local(torch.ones(2, 3) * 999)
        self.assertTrue(torch.any(result > 100))
        for key, value in saved.items():
            self.assertTrue(torch.equal(value, local.state_dict()[key]))

    def test_scaled_noise_and_noise_free_critic_inputs(self):
        components = [torch.zeros(4, n) for n in (3, 3, 3, 4, 12, 12, 12)]
        components[0].fill_(1)
        components[5].fill_(10)
        seed = 47
        actor = build_actor_observation(*components, add_noise=True, generator=torch.Generator().manual_seed(seed))
        clean = build_actor_observation(*components, add_noise=False)
        random = 2 * torch.rand((4,48), generator=torch.Generator().manual_seed(seed)) - 1
        noise = random * torch.tensor(STAGE1_CONFIG.observation.noise_half_ranges_after_scaling)
        self.assertTrue(torch.equal(actor, clean + noise))
        self.assertTrue(torch.equal(clean[:,24:36], torch.full((4,12), .5)))
        critic = build_critic_observation(clean, torch.zeros(4,3), torch.zeros(4,3), torch.ones(4), torch.zeros(4,4), torch.full((4,294),2.))
        self.assertTrue(torch.equal(critic[:,:48], clean))
        self.assertTrue(torch.equal(critic[:,-294:], torch.full((4,294),5.)))
        # Environment clipping is independent of the unbounded normalizer output.
        components[0].fill_(1000)
        clipped = build_actor_observation(*components, add_noise=True, generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(clipped[:,:3], torch.full((4,3),100.)))

    def test_ppo_cache_update_counts_and_fresh_process_restore(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            runner = make_runner(folder)
            self.assertIs(type(runner.alg), PPO)
            optimizer_ids = {id(p) for group in runner.alg.optimizer.param_groups for p in group['params']}
            self.assertEqual(optimizer_ids, {id(p) for p in runner.alg.actor_critic.parameters()})
            obs = runner.env.get_observations()
            critic = runner.env.get_privileged_observations()
            runner.env.get_observations()
            runner.env.get_privileged_observations()
            self.assertEqual(runner.normalizers.actor.count.item(), 4)
            self.assertEqual(runner.normalizers.critic.count.item(), 4)
            for _ in range(4):
                old_obs, old_critic = obs.clone(), critic.clone()
                with torch.inference_mode():
                    actions = runner.alg.act(obs, critic)
                    obs, critic, rewards, dones, infos = runner.env.step(actions)
                    runner.alg.process_env_step(rewards, dones, infos)
                step = runner.alg.storage.step - 1
                self.assertTrue(torch.equal(runner.alg.storage.observations[step], old_obs))
                self.assertTrue(torch.equal(runner.alg.storage.privileged_observations[step], old_critic))
            self.assertEqual(runner.normalizers.actor.count.item(), 20)
            self.assertEqual(runner.normalizers.critic.count.item(), 20)
            saved = copy.deepcopy(runner.normalizers.state_dict())
            with torch.inference_mode():
                runner.alg.compute_returns(critic)
            losses = runner.alg.update()
            self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
            for key, value in saved.items():
                self.assertTrue(torch.equal(value, runner.normalizers.state_dict()[key]))
            runner.current_learning_iteration = 1
            probe = {'actor': obs.clone(), 'critic': critic.clone()}
            expected = outputs(runner, probe['actor'], probe['critic'])
            runner.save(folder / 'policy.pt')
            torch.save(probe, folder / 'sample.pt')
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--restore', str(folder)], env=env, check=True, timeout=60, capture_output=True)
            restored = torch.load(folder / 'restored.pt')
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, restored['outputs'][key]), key)
            self.assertEqual(restored['iteration'], 1)
            self.assertEqual(restored['learning_rate'], runner.alg.learning_rate)
            self.assert_nested_equal(runner.alg.optimizer.state_dict(), restored['optimizer'])
            self.assert_nested_equal(runner.alg.actor_critic.state_dict(), restored['model'])
            # Missing statistics must be rejected even if other metadata is intact.
            broken = torch.load(folder / 'policy.pt')
            del broken['model_state_dict']['actor.0.count']
            torch.save(broken, folder / 'broken.pt')
            with self.assertRaisesRegex(RuntimeError, 'missing normalizer statistics'):
                runner.load(folder / 'broken.pt')
            del broken['observation_normalization']
            torch.save(broken, folder / 'legacy.pt')
            with self.assertRaisesRegex(RuntimeError, 'normalization semantics'):
                runner.load(folder / 'legacy.pt')

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a: self.assert_nested_equal(a[key],b[key])
        elif isinstance(a, (tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b): self.assert_nested_equal(x,y)
        else:
            self.assertEqual(a,b)

    def test_inherited_learning_loop_and_production_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(directory)
            runner.get_inference_policy()  # eval -> train transition must count the initial batch
            runner.learn(1, init_at_random_ep_len=True)
            self.assertIs(runner.env.episode_length_buf, runner.env.env.episode_length_buf)
            self.assertTrue(torch.any(runner.env.env.episode_length_buf > 0))
            self.assertEqual(runner.normalizers.actor.count.item(), 20)
            self.assertEqual(runner.normalizers.critic.count.item(), 20)
            with self.assertRaisesRegex(ValueError, 'requires the PACE v2 validation environment'):
                PaceV2ValidationRunner(TensorEnv(), ppo_train_cfg())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--restore':
        worker(sys.argv[2])
    else:
        unittest.main()
