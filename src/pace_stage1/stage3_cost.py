"""Explicit Stage3 candidate cost units; no reward shaping or budget selection."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class EnergyCostSpec:
    # No default: callers must state which scientific quantity they constrain.
    definition: str
    policy_dt_s: float = .01

    def __post_init__(self):
        if self.definition not in ('diagnostic_nonnegative', 'pace_corrected'):
            raise ValueError('explicit supported energy cost definition required')
        if not math.isfinite(self.policy_dt_s) or self.policy_dt_s<=0:
            raise ValueError('positive finite policy timestep required')

    def power_w(self, mean_components_w, commands):
        """Components are substep means: electrical, positive net mechanical, potential."""
        if mean_components_w.shape[-1]!=3 or not torch.isfinite(mean_components_w).all():
            raise ValueError('finite electrical/mechanical/potential components required')
        if self.definition=='diagnostic_nonnegative':
            value=mean_components_w[...,:2].sum(-1)
            if (mean_components_w[...,:2]<0).any():
                raise ValueError('diagnostic physical components must be nonnegative')
            return value
        if not torch.isfinite(commands[...,:2]).all():
            raise ValueError('finite planar commands required')
        return mean_components_w.sum(-1)/(1.+commands[...,:2].square().sum(-1))

    def step_cost_j(self, mean_components_w, commands):
        return self.policy_dt_s*self.power_w(mean_components_w,commands)


class EnergyCostLedger:
    """Accumulate all transition energy, including terminal steps, before reset."""
    def __init__(self,num_envs,device,policy_dt_s=.01):
        self.dt=policy_dt_s
        self.energy_j=torch.zeros(num_envs,device=device,dtype=torch.float64)
        self.steps=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.total_energy_j=torch.zeros((),device=device,dtype=torch.float64)
        self.total_steps=0

    def add(self,step_cost_j):
        if step_cost_j.shape!=self.energy_j.shape or not torch.isfinite(step_cost_j).all():
            raise ValueError('finite scalar energy per environment required')
        values=step_cost_j.detach().to(torch.float64)
        self.energy_j+=values;self.steps+=1
        self.total_energy_j+=values.sum();self.total_steps+=values.numel()

    def finish(self,ids):
        duration=self.steps[ids].to(torch.float64)*self.dt
        energy=self.energy_j[ids].clone()
        record=dict(env_ids=ids.clone(),energy_j=energy,duration_s=duration,
                    mean_power_w=energy/duration.clamp_min(self.dt))
        self.energy_j[ids]=0.;self.steps[ids]=0
        return record

    def mean_power_w(self):
        return self.total_energy_j/(max(1,self.total_steps)*self.dt)
