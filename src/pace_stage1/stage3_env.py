"""Engineering-only Stage3 environment: inherited task reward plus separate cost."""
# Import simulator before torch, as required by Isaac Gym.
from .pace_v2_formal import FormalEnv
from .construction_staging import CapacitySafeConstructionMixin
from .stage2_energy import energy_components
from .stage3_cost import EnergyCostSpec, EnergyCostLedger
import torch


class Stage3CostEnv(CapacitySafeConstructionMixin,FormalEnv):
    """Task reward with signed cost consumed independently by the Stage3 runner."""
    def __init__(self,*,cost_definition,**kwargs):
        self._cost_step_active=False
        super().__init__(**kwargs)
        self.cost_spec=EnergyCostSpec(cost_definition,self.policy_dt)
        self.cost_ledger=EnergyCostLedger(self.num_envs,self.device,self.policy_dt)
        # Keep this engineering interface out of the existing formal PPO entry.
        from types import SimpleNamespace
        self.protocol_profile=SimpleNamespace(identity='PACE_V2_STAGE3_COST_ENGINEERING')

    def _step_actuator(self,target):
        result=super()._step_actuator(target)
        if self._cost_step_active:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._cost_components+=energy_components(result.applied_torque,self.dof_vel,
                self.body_mass,self.rigid_body_state[:,:,9],potential_sign=-1)/self.cfg.action.policy_decimation
        return result

    def _reset_idx(self,ids):
        if self._cost_step_active:
            # Same pre-reset command tensor used by Stage2 energy reward, including
            # the base environment's scheduled command update on this transition.
            self._cost_commands=self.commands.clone()
            self._cost_power=self.cost_spec.power_w(self._cost_components,self._cost_commands)
            self._step_cost=self._cost_power*self.policy_dt
            self.cost_ledger.add(self._step_cost)
            self._cost_completed=self.cost_ledger.finish(ids)
        elif hasattr(self,'cost_ledger'):
            self.cost_ledger.finish(ids)
        super()._reset_idx(ids)

    def step(self,actions):
        self._cost_components=torch.zeros((self.num_envs,3),device=self.device)
        self._cost_step_active=True
        try:
            obs,critic,reward,done,infos=super().step(actions)
        finally:
            self._cost_step_active=False
        infos['costs']=self._step_cost
        infos['cost_power_w']=self._cost_power
        infos['cost_components_w']=self._cost_components
        infos['cost_commands']=self._cost_commands
        infos['cost_episodes']=self._cost_completed
        infos['cost_definition']=self.cost_spec.definition
        return obs,critic,reward,done,infos
