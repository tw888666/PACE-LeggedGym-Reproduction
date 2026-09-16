import copy
import unittest

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from pace_stage1.stage3_lagrangian import (
    ConstraintConfig, LagrangianPPO, differential_cost_returns, projected_multiplier,
)


def model():
    policy = ActorCritic(2, 3, 1, actor_hidden_dims=[8], critic_hidden_dims=[8],
                         activation='elu', init_noise_std=1.)
    policy.critic = nn.Sequential(nn.Identity(), policy.critic)
    return policy


def algorithm(policy=None, **config):
    return LagrangianPPO(policy or model(),
        constraint=ConstraintConfig(reference_power_w=100., **config),
        critic_normalizer=nn.Identity(), num_learning_epochs=2, num_mini_batches=2,
        learning_rate=.01, gamma=.99, lam=.95, entropy_coef=.001,
        use_clipped_value_loss=False)


def collect(alg, powers=None):
    alg.init_storage(4, 3, [2], [3], [1])
    with torch.inference_mode():
        obs, critic = torch.randn(4, 2), torch.randn(4, 3)
        for i in range(3):
            alg.act(obs, critic)
            rewards = torch.tensor([.1, -.2, .3, .4])
            done = torch.tensor([i == 1, i == 2, False, False])
            infos = dict(costs=.01*torch.tensor(powers or [-20., 10., 200., 400.]),
                         cost_definition='pace_corrected',
                         time_outs=torch.tensor([False, i == 2, False, False]))
            alg.process_env_step(rewards, done, infos)
            obs, critic = torch.randn(4, 2), torch.randn(4, 3)
        alg.compute_returns(critic)


class Stage3LagrangianChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_average_cost_units_signed_returns_and_rollout_bootstrap(self):
        powers = torch.tensor([1., -1., 2.]).reshape(3, 1, 1)
        values = torch.tensor([.5, 1., -.5]).reshape(3, 1, 1)
        returns, advantages = differential_cost_returns(powers, values, torch.tensor([[2.]]), 1.)
        torch.testing.assert_close(returns.flatten(), torch.tensor([2., 5/3, 10/3]))
        torch.testing.assert_close(advantages, returns-values)
        negative, _ = differential_cost_returns(torch.tensor([-3., -1.]).reshape(2, 1, 1),
            torch.zeros(2, 1, 1), torch.zeros(1, 1), 0.)
        torch.testing.assert_close(negative.flatten(), torch.tensor([-1., 1.]))

    def test_dual_sign_budget_equality_and_zero_projection(self):
        cfg = ConstraintConfig(100., dual_learning_rate=.1)
        self.assertAlmostEqual(projected_multiplier(1., 190., cfg), 1.1)
        self.assertAlmostEqual(projected_multiplier(1., 90., cfg), 1.)
        self.assertAlmostEqual(projected_multiplier(1., -10., cfg), .9)
        self.assertEqual(projected_multiplier(0., -100., cfg), 0.)

    def test_zero_constraint_matches_original_ppo_including_task_timeouts(self):
        original = model()
        constrained = algorithm(copy.deepcopy(original), budget_fraction=100.)
        baseline = PPO(original, num_learning_epochs=2, num_mini_batches=2,
                       learning_rate=.01, gamma=.99, lam=.95, entropy_coef=.001,
                       use_clipped_value_loss=False)
        rng = torch.get_rng_state()
        collect(baseline)
        torch.set_rng_state(rng)
        collect(constrained)
        torch.testing.assert_close(constrained.storage.rewards, baseline.storage.rewards, rtol=0, atol=0)
        torch.testing.assert_close(constrained.storage.returns, baseline.storage.returns, rtol=0, atol=0)
        rng = torch.get_rng_state()
        expected = baseline.update()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = constrained.update()
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(constrained.actor_critic.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(constrained.multiplier, 0.)

    def test_cost_process_continues_across_failure_and_timeout(self):
        alg = algorithm()
        collect(alg)
        # Both reset types remain ordinary transitions in the average-cost chain.
        self.assertTrue(alg.storage.dones[1, 0])
        self.assertTrue(alg.storage.dones[2, 1])
        expected_delta = (alg.cost_power_ratio[1, 0]-alg.cost_power_ratio.mean()
                          +alg.cost_values[2, 0]-alg.cost_values[1, 0])
        torch.testing.assert_close(alg.cost_advantages[1, 0],
            expected_delta+alg.constraint.cost_trace_lambda*alg.cost_advantages[2, 0])
        self.assertAlmostEqual(alg.cost_power_ratio[0, 0].item(), -.2, places=6)

    def test_active_constraint_reduces_high_cost_action_probability(self):
        alg = algorithm(initial_multiplier=1., cost_trace_lambda=0.)
        for p in alg.cost_critic.parameters():
            nn.init.zeros_(p)
        alg.init_storage(512, 1, [2], [3], [1])
        obs, critic = torch.zeros(512, 2), torch.zeros(512, 3)
        with torch.inference_mode():
            actions = alg.act(obs, critic)
            before = alg.actor_critic.action_mean[0, 0].item()
            # Larger sampled actions incur larger power, independent of reward.
            powers = 100.+100.*(actions[:, 0] > before).float()
            alg.process_env_step(torch.zeros(512), torch.zeros(512, dtype=torch.bool),
                dict(costs=.01*powers, cost_definition='pace_corrected'))
            alg.compute_returns(critic)
        alg.update()
        with torch.no_grad():
            after = alg.actor_critic.act_inference(obs)[0, 0].item()
        self.assertLess(after, before)
        self.assertGreater(alg.multiplier, 1.)

    def test_checkpoint_restores_cost_optimizer_and_next_update(self):
        alg = algorithm(initial_multiplier=.5)
        collect(alg); alg.update()
        state = copy.deepcopy(alg.constraint_state_dict())
        restored_model = model()
        restored_model.load_state_dict(alg.actor_critic.state_dict())
        restored = algorithm(restored_model, initial_multiplier=.5)
        restored.optimizer.load_state_dict(copy.deepcopy(alg.optimizer.state_dict()))
        restored.learning_rate = alg.learning_rate
        restored.load_constraint_state_dict(state)
        self.assertEqual(restored.multiplier, alg.multiplier)
        rng = torch.get_rng_state()
        collect(alg); alg.update()
        torch.set_rng_state(rng)
        collect(restored); restored.update()
        self.assertEqual(restored.multiplier, alg.multiplier)
        for a, b in zip(alg.cost_critic.parameters(), restored.cost_critic.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(alg.actor_critic.parameters(), restored.actor_critic.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        other = algorithm(budget_fraction=.8)
        with self.assertRaises(ValueError):
            other.load_constraint_state_dict(state)

    def test_uneven_minibatches_keep_baseline_batch_count(self):
        alg = algorithm()
        alg.num_mini_batches = 10
        collect(alg)
        alg.update()
        self.assertTrue(all(state['step'] == 20 for state in alg.cost_optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
