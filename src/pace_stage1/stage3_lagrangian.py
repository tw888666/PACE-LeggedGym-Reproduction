"""Stage3 average-power constraint; task PPO update remains rsl_rl v1.0.2.

The cost process continues through automatic resets (including time limits).
Its differential value predicts future excess power over the rollout mean.
This is an engineering PPO-Lagrangian implementation, not paper-exact ECO.
"""
import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from rsl_rl.algorithms import PPO


@dataclass(frozen=True)
class ConstraintConfig:
    reference_power_w: float
    budget_fraction: float = 0.9
    dual_learning_rate: float = 0.01
    initial_multiplier: float = 0.0
    cost_learning_rate: float = 0.001
    cost_trace_lambda: float = 0.95
    policy_dt_s: float = 0.01
    cost_definition: str = 'pace_corrected'
    estimator: str = 'continuing_differential_gae_v1'

    def __post_init__(self):
        for key in ('reference_power_w', 'budget_fraction', 'dual_learning_rate',
                    'cost_learning_rate', 'policy_dt_s'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite ' + key + ' required')
        if not math.isfinite(self.initial_multiplier) or self.initial_multiplier < 0:
            raise ValueError('nonnegative finite multiplier required')
        if not 0 <= self.cost_trace_lambda <= 1:
            raise ValueError('cost trace lambda must be in [0, 1]')
        if self.cost_definition != 'pace_corrected' or self.estimator != 'continuing_differential_gae_v1':
            raise ValueError('unsupported Stage3 cost semantics')

    @property
    def budget_power_w(self):
        return self.reference_power_w * self.budget_fraction


def differential_cost_returns(power_ratio, values, last_values, trace_lambda):
    """Undiscounted differential GAE, including reset-state bootstrap.

    All entries have equal policy dt. No done mask: the constrained process
    includes every real simulated step, with automatic reset as a transition.
    A rollout boundary bootstraps but does not truncate the physical cost.
    """
    mean = power_ratio.mean()
    advantage = torch.zeros_like(last_values)
    advantages = torch.empty_like(values)
    for step in reversed(range(len(values))):
        next_value = last_values if step == len(values)-1 else values[step+1]
        delta = power_ratio[step] - mean + next_value - values[step]
        advantage = delta + trace_lambda * advantage
        advantages[step] = advantage
    return advantages + values, advantages


def projected_multiplier(multiplier, mean_power_w, config):
    violation = mean_power_w / config.reference_power_w - config.budget_fraction
    value = max(0.0, multiplier + config.dual_learning_rate * violation)
    if not math.isfinite(value) or not math.isfinite(violation):
        raise RuntimeError('non-finite multiplier or cost violation')
    return value


class LagrangianPPO(PPO):
    def __init__(self, actor_critic, *, constraint, critic_normalizer, **kwargs):
        super().__init__(actor_critic, **kwargs)
        if actor_critic.is_recurrent:
            raise ValueError('Stage3 supports the existing feed-forward policy only')
        self.constraint = constraint
        self.critic_normalizer = critic_normalizer
        # Independent parameters, identical architecture and initial values.
        # Copying consumes no RNG and preserves the baseline actor initialization.
        self.cost_critic = copy.deepcopy(actor_critic.critic[1]).to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                                               lr=constraint.cost_learning_rate)
        self.cost_rng = torch.Generator(device=self.device).manual_seed(0)
        self.multiplier = constraint.initial_multiplier
        self.last_constraint_metrics = {}

    def evaluate_cost(self, observations):
        # Reuse already-counted critic input statistics; never count twice.
        return self.cost_critic(self.critic_normalizer(observations))

    def init_storage(self, *args, **kwargs):
        super().init_storage(*args, **kwargs)
        self.cost_values = torch.zeros_like(self.storage.values)
        self.cost_power_ratio = torch.zeros_like(self.storage.values)

    def act(self, obs, critic_obs):
        self.cost_values[self.storage.step].copy_(self.evaluate_cost(critic_obs).detach())
        return super().act(obs, critic_obs)

    def process_env_step(self, rewards, dones, infos):
        if infos.get('cost_definition') != self.constraint.cost_definition:
            raise ValueError('environment/constraint cost definition mismatch')
        costs = infos['costs'].to(self.device)
        if not torch.isfinite(costs).all():
            raise RuntimeError('non-finite cost')
        self.cost_power_ratio[self.storage.step].copy_(costs.reshape(-1, 1) /
            (self.constraint.policy_dt_s * self.constraint.reference_power_w))
        # Parent retains task reward, timeout bootstrap, and reward GAE unchanged.
        super().process_env_step(rewards, dones, infos)

    def compute_returns(self, last_critic_obs):
        super().compute_returns(last_critic_obs)
        last_values = self.evaluate_cost(last_critic_obs).detach()
        self.cost_returns, self.cost_advantages = differential_cost_returns(
            self.cost_power_ratio, self.cost_values, last_values,
            self.constraint.cost_trace_lambda)

    def update(self):
        cfg = self.constraint
        mean_power = self.cost_power_ratio.double().mean().item() * cfg.reference_power_w
        previous_multiplier = self.multiplier
        self.multiplier = projected_multiplier(self.multiplier, mean_power, cfg)
        # Parent reward advantages were normalized by the unchanged baseline.
        # Cost units are fixed by B_ref, not divided by a changing cost std.
        cost_adv = self.cost_advantages - self.cost_advantages.mean()
        self.storage.advantages = self.storage.advantages - self.multiplier * cost_adv
        observations = self.storage.privileged_observations.flatten(0, 1)
        targets = self.cost_returns.flatten(0, 1).clone()
        old_values = self.cost_values.flatten(0, 1)
        with torch.no_grad():
            variance = targets.var(unbiased=False)
            explained = (1 - (targets-old_values).var(unbiased=False)/variance).item() if variance > 1e-12 else None
        # Uses exactly the baseline clipping, adaptive KL, reward-value loss,
        # entropy, minibatches, epochs, optimizer and gradient clipping.
        losses = super().update()
        size = len(observations)
        indices = torch.randperm(size, device=self.device, generator=self.cost_rng)
        cost_loss = 0.0
        self.cost_critic.train()
        for _ in range(self.num_learning_epochs):
            for batch in torch.tensor_split(indices, self.num_mini_batches):
                predicted = self.evaluate_cost(observations[batch])
                target = targets[batch]
                if self.use_clipped_value_loss:
                    clipped = old_values[batch] + (predicted-old_values[batch]).clamp(-self.clip_param, self.clip_param)
                    loss = torch.maximum((predicted-target).square(), (clipped-target).square()).mean()
                else:
                    loss = (predicted-target).square().mean()
                self.cost_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_optimizer.step()
                cost_loss += loss.item()
        cost_loss /= self.num_learning_epochs * self.num_mini_batches
        if not math.isfinite(cost_loss) or not all(torch.isfinite(p).all() for p in self.cost_critic.parameters()):
            raise RuntimeError('non-finite cost critic update')
        self.last_constraint_metrics = dict(
            mean_power_w=mean_power, budget_power_w=cfg.budget_power_w,
            violation_w=mean_power-cfg.budget_power_w,
            relative_violation=mean_power/cfg.reference_power_w-cfg.budget_fraction,
            multiplier_before=previous_multiplier, multiplier=self.multiplier,
            cost_value_loss=cost_loss, cost_explained_variance=explained,
            cost_return_mean=self.cost_returns.mean().item(),
            cost_return_std=self.cost_returns.std().item(),
            cost_advantage_std=self.cost_advantages.std().item(),
            negative_step_fraction=(self.cost_power_ratio < 0).float().mean().item(),
            task_learning_rate=self.learning_rate)
        return losses

    def constraint_state_dict(self):
        return dict(config=asdict(self.constraint), multiplier=self.multiplier,
                    cost_critic=self.cost_critic.state_dict(),
                    cost_optimizer=self.cost_optimizer.state_dict(),
                    cost_rng=self.cost_rng.get_state())

    def load_constraint_state_dict(self, state, load_optimizer=True):
        if state['config'] != asdict(self.constraint):
            raise ValueError('checkpoint constraint/budget differs from runtime')
        multiplier = float(state['multiplier'])
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError('invalid checkpoint multiplier')
        self.cost_critic.load_state_dict(state['cost_critic'])
        if load_optimizer:
            self.cost_optimizer.load_state_dict(state['cost_optimizer'])
        self.cost_rng.set_state(state['cost_rng'].cpu())
        self.multiplier = multiplier
