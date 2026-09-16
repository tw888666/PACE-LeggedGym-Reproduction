"""Stage3 runner with independent cost state and the existing task loop."""
from dataclasses import asdict

import torch
from rsl_rl.runners import OnPolicyRunner

from .normalization_bridge import NormalizationRunnerMixin
from .pace_v2_runner import ValidationLoopMixin
from .stage3_lagrangian import LagrangianPPO


class Stage3Runner(ValidationLoopMixin, NormalizationRunnerMixin, OnPolicyRunner):
    def __init__(self, env, train_cfg, *, constraint, log_dir=None, device='cpu'):
        if env.cost_spec.definition != constraint.cost_definition or env.policy_dt != constraint.policy_dt_s:
            raise ValueError('environment cost units differ from constraint')
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.alg = LagrangianPPO(self.alg.actor_critic, constraint=constraint,
            critic_normalizer=self.normalizers.critic, device=device, **self.alg_cfg)
        self.alg.init_storage(env.num_envs, self.num_steps_per_env,
                             [env.num_obs], [env.num_privileged_obs], [env.num_actions])

    def iteration(self):
        row = super().iteration()
        row['constraint'] = self.alg.last_constraint_metrics
        return row

    def save(self, path, infos=None):
        torch.save(dict(
            model_state_dict=self.alg.actor_critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            iter=self.current_learning_iteration, infos=dict(infos or {}, stage='Stage3 average-power constraint',
                profile='PACE_V2_STAGE3', training_setup=getattr(self, 'training_setup', None)),
            observation_normalization=self._normalization_metadata(),
            entropy_schedule=self.entropy_scheduler.state_dict(),
            iteration_semantics='next_update_index',
            stage3_constraint=self.alg.constraint_state_dict()), path)

    def load(self, path, load_optimizer=True):
        payload = torch.load(path, map_location=self.device)
        constraint = payload.get('stage3_constraint')
        if constraint is None or constraint['config'] != asdict(self.alg.constraint):
            raise ValueError('resume requires a Stage3 checkpoint with matching cost and budget')
        # The baseline normalizer recomputes _std during inference-mode rollout.
        # Make those buffers writable for an in-process checkpoint reload.
        for normalizer in (self.normalizers.actor, self.normalizers.critic):
            for name, value in normalizer._buffers.items():
                if value is not None and value.is_inference():
                    normalizer._buffers[name] = value.clone()
        result = super().load(path, load_optimizer)
        self.alg.load_constraint_state_dict(constraint, load_optimizer)
        return result
