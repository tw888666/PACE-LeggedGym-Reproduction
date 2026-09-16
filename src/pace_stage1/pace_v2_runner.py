"""Explicit iteration clock for validation; PPO update implementation stays v1.0.2."""
import time
import torch

from .config import ppo_train_cfg
from .pace_v2_profiles import PaceV2PPOConfig
from .pace_v2_entropy import PaceV2EntropyScheduler
from .normalization_bridge import PaceV2ValidationRunner


def validation_train_cfg(iterations):
    p = PaceV2PPOConfig()
    cfg = ppo_train_cfg()
    cfg['policy'].update(actor_hidden_dims=list(p.actor_hidden_dims), critic_hidden_dims=list(p.critic_hidden_dims), init_noise_std=p.init_noise_std, activation=p.activation)
    for key in ('value_loss_coef','use_clipped_value_loss','clip_param','num_learning_epochs','num_mini_batches','learning_rate','schedule','gamma','lam','desired_kl','max_grad_norm'):
        cfg['algorithm'][key] = getattr(p,key)
    cfg['algorithm']['entropy_coef'] = p.entropy.initial
    cfg['runner'].update(num_steps_per_env=p.num_steps_per_env,max_iterations=iterations)
    return cfg


class ValidationLoopMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_scheduler = PaceV2EntropyScheduler(PaceV2PPOConfig().entropy)

    def save(self, path, infos=None):
        torch.save({'model_state_dict':self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict':self.alg.optimizer.state_dict(),
                    'iter':self.current_learning_iteration, 'infos':infos,
                    'observation_normalization':self._normalization_metadata(),
                    'entropy_schedule':self.entropy_scheduler.state_dict(),
                    'iteration_semantics':'next_update_index'}, path)

    def load(self, path, load_optimizer=True):
        payload = torch.load(path,map_location=self.device)
        if payload.get('iteration_semantics') != 'next_update_index':
            raise RuntimeError('checkpoint has no explicit next-update clock')
        schedule = payload['entropy_schedule']
        if schedule['last_iteration'] != payload['iter']-1:
            raise RuntimeError('checkpoint entropy/next-update iteration mismatch')
        self.entropy_scheduler.load_state_dict(schedule)
        return super().load(path,load_optimizer)

    def iteration(self):
        i = self.current_learning_iteration
        self.alg.actor_critic.train()
        self.env.set_training_iteration(i)
        entropy = self.entropy_scheduler.apply(self.alg,i)
        obs = self.env.get_observations().to(self.device)
        critic = self.env.get_privileged_observations().to(self.device)
        sums = dict(reward=0.,raw_action_abs=0.,raw_action_max=0.,policy_mean_abs=0.,policy_mean_max=0.)
        episodes=[]
        start=time.monotonic()
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                actions=self.alg.act(obs,critic)
                mean=self.alg.actor_critic.action_mean
                sums['raw_action_abs'] += actions.abs().mean().item()
                sums['raw_action_max'] = max(sums['raw_action_max'],actions.abs().max().item())
                sums['policy_mean_abs'] += mean.abs().mean().item()
                sums['policy_mean_max'] = max(sums['policy_mean_max'],mean.abs().max().item())
                obs,critic,rewards,dones,infos=self.env.step(actions.to(self.env.device))
                obs,critic,rewards,dones=obs.to(self.device),critic.to(self.device),rewards.to(self.device),dones.to(self.device)
                if not all(torch.isfinite(x).all() for x in (obs,critic,rewards,actions)):
                    raise RuntimeError('non-finite observation/reward/action')
                sums['reward'] += rewards.mean().item()
                if infos.get('episode'): episodes.append(infos['episode'])
                self.alg.process_env_step(rewards,dones,infos)
            self.alg.compute_returns(critic)
        losses=self.alg.update()
        if not all(torch.isfinite(p).all() for p in self.alg.actor_critic.parameters()):
            raise RuntimeError('non-finite policy parameters')
        if not all(torch.isfinite(torch.tensor(v)) for v in losses):
            raise RuntimeError('non-finite PPO loss')
        self.current_learning_iteration=i+1
        for key in ('reward','raw_action_abs','policy_mean_abs'): sums[key]/=self.num_steps_per_env
        std=self.alg.actor_critic.std.detach()
        sums.update(iteration=i,next_iteration=i+1,entropy_coef=entropy,value_loss=losses[0],surrogate_loss=losses[1],std_mean=std.mean().item(),std_max=std.max().item(),actor_count=self.normalizers.actor.count.item(),critic_count=self.normalizers.critic.count.item(),seconds=time.monotonic()-start)
        if torch.any(std<=0): raise RuntimeError('nonpositive policy std')
        if episodes:
            sums['episode']={k:sum(float(e[k]) for e in episodes if k in e)/sum(k in e for e in episodes) for k in set().union(*(e.keys() for e in episodes))}
        return sums


class ValidationRunner(ValidationLoopMixin,PaceV2ValidationRunner):
    pass
