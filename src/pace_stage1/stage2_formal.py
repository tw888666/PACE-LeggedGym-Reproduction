"""Stage2-only energy addition; the frozen Stage1 implementation is inherited."""
from .pace_v2_formal import FormalEnv, FormalRunner, save_verified, MILESTONES
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
import torch
from .config import STAGE1_CONFIG
from .formal_diagnostics import normalization_summary
from .pace_v2_rewards import pace_v2_environment_config
from .pace_v2_runner import validation_train_cfg
from .pace_v2_validation import write
from .protocol_gate import require_formal_training_allowed, PROJECT_ROOT
from .stage2_energy import energy_components, energy_reward


class Stage2Env(FormalEnv):
    def __init__(self, *, potential_sign, **kwargs):
        if potential_sign not in (-1,1):raise ValueError('explicit potential sign required')
        self.potential_sign=potential_sign
        self._energy_step_active=False
        super().__init__(**kwargs)
        self.episode_sums['energy']=torch.zeros(self.num_envs,device=self.device)

    def reset_metrics(self):
        super().reset_metrics()
        self.stage2_power_sum=torch.zeros(3,device=self.device)
        self.stage2_reward_sum=0.
        self.stage2_steps=0

    def _step_actuator(self,target):
        result=super()._step_actuator(target)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        power=energy_components(result.applied_torque,self.dof_vel,self.body_mass,
                                self.rigid_body_state[:,:,9],potential_sign=self.potential_sign)
        self._energy_power+=power/self.cfg.action.policy_decimation
        return result

    def _reset_idx(self,ids):
        if self._energy_step_active:
            # The base implementation calls this exactly once after task rewards,
            # even for empty ids. Capture reward commands before reset resampling.
            self._energy_delta,self._energy_unscaled,self._energy_gamma=energy_reward(
                self._energy_power,self.commands,self.training_iteration,self.policy_dt)
            self.episode_sums['energy']+=self._energy_delta
            self.episode_sums['total']+=self._energy_delta
        super()._reset_idx(ids)

    def step(self,actions):
        self._energy_power=torch.zeros((self.num_envs,3),device=self.device)
        self._energy_step_active=True
        try:
            obs,critic,task,done,infos=super().step(actions)
        finally:
            self._energy_step_active=False
        # Inherited Stage1 instrumentation has already checked the task term.
        total=task+self._energy_delta
        if not torch.isfinite(total).all():raise RuntimeError('non-finite Stage2 reward')
        self.rew_buf=total
        infos['reward_terms'].update(energy=self._energy_unscaled,
                                     scaled_energy=self._energy_delta,
                                     energy_gamma_v=self._energy_gamma)
        self.stage2_power_sum+=self._energy_power.mean(0)
        self.stage2_reward_sum+=self._energy_delta.mean().item()
        self.stage2_steps+=1
        return obs,critic,total,done,infos

    def metrics(self):
        result=super().metrics()
        result['stage2_energy']=dict(mean_power_components_w=(self.stage2_power_sum/max(1,self.stage2_steps)).cpu().tolist(),
                                    mean_scaled_reward=self.stage2_reward_sum/max(1,self.stage2_steps),
                                    potential_sign=self.potential_sign)
        return result


class Stage2Runner(FormalRunner):
    def save(self,path,infos=None):
        metadata=dict(infos or {})
        metadata.update(stage='Stage2 PACE energy curriculum',potential_sign=self.env.potential_sign,
                        energy_scale=-16e-5,energy_half_life=500.)
        return super().save(path,metadata)

    def load(self,path,load_optimizer=True):
        saved=torch.load(path,map_location='cpu').get('infos',{})
        if saved.get('stage')!='Stage2 PACE energy curriculum' or saved.get('potential_sign')!=self.env.potential_sign:
            raise RuntimeError('resume must use a matching Stage2 checkpoint, never Stage1 warm start')
        return super().load(path,load_optimizer)


def run(output,potential_sign,resume=None,construction_capacity_repair=False):
    require_formal_training_allowed()
    output.mkdir(parents=True,exist_ok=resume is not None)
    cfg=validation_train_cfg(30000);cfg['seed']=0
    cfg['runner']['experiment_name']='pace_v2_formal_stage2_energy_curriculum'
    manifest=dict(stage='Stage2 PACE energy curriculum',classification='PACE_V2_FORMAL / RECONSTRUCTION',
                  state='constructing',num_envs=4096,seed=0,iterations=30000,terrain='trimesh',
                  train_cfg=cfg,environment_config=asdict(pace_v2_environment_config(STAGE1_CONFIG)),
                  energy=dict(scale=-16e-5,half_life=500.,potential_sign=potential_sign,
                              potential_sign_paper_exact=False,substep_mean_power=True),
                  initialization='fresh seed0; no Stage1 warm start',started_unix=time.time(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if construction_capacity_repair:
        manifest['physics_construction_repair']=dict(separate_construction_only=True,max_gpu_contact_pairs=2**25,
                                                   default_buffer_size_multiplier=5.,changes_learning_protocol=False)
        if resume is not None and json.loads((output/'gpt-运行记录.json').read_text()).get('physics_construction_repair')!=manifest['physics_construction_repair']:
            raise ValueError('cannot resume an unrepaired run into the repaired protocol')
    if resume is not None:
        manifest.update(resume_from=str(resume),resume_limitation='simulator and RNG reset; not bitwise trajectory continuation')
    env=None
    try:
        snapshot=output/('gpt-source' if resume is None else 'gpt-source-resume-%d'%int(time.time()))
        shutil.copytree(PROJECT_ROOT/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(PROJECT_ROOT/'provenance',snapshot/'provenance')
        write(output/'gpt-运行记录.json',manifest)
        torch.set_num_threads(4)
        env_class=Stage2Env
        if construction_capacity_repair:
            from .construction_staging import CapacitySafeConstructionMixin
            class RepairedEnv(CapacitySafeConstructionMixin,Stage2Env):pass
            env_class=RepairedEnv
        env=env_class(potential_sign=potential_sign)
        runner=Stage2Runner(env,cfg,log_dir=str(output),device='cuda:0')
        if resume is not None:runner.load(resume)
        start=runner.current_learning_iteration
        initial_count=runner.normalizers.actor.count.item()
        save_verified(runner,output/('gpt-model-%d.pt'%start))
        manifest.update(state='running',segment_start_iteration=start,gpu_name=torch.cuda.get_device_name(0))
        write(output/'gpt-运行记录.json',manifest)
        with (output/'gpt-逐迭代指标.jsonl').open('a' if resume else 'w') as stream:
            for i in range(start,30000):
                env.reset_metrics();row=runner.iteration();row.update(env.metrics())
                row['normalization']=normalization_summary(runner.normalizers)
                if row['entropy_coef']!=runner.entropy_scheduler.coefficient(i):raise RuntimeError('entropy mismatch')
                if row['actor_count']!=initial_count+4096*(1+24*(i-start+1)) or row['critic_count']!=row['actor_count']:
                    raise RuntimeError('normalizer count mismatch')
                if env.eq9_calls!=(i-start+1)*24*4:raise RuntimeError('Eq9 count mismatch')
                line=json.dumps(row,allow_nan=False);stream.write(line+'\n');stream.flush();print(line,flush=True)
                if (i+1)%500==0 or (i+1) in MILESTONES:
                    save_verified(runner,output/('gpt-model-%d.pt'%(i+1)))
                    manifest['completed_iterations']=i+1;write(output/'gpt-运行记录.json',manifest)
        shutil.copyfile(output/'gpt-model-30000.pt',output/'gpt-model-final.pt')
        manifest.update(state='completed',completed_iterations=30000)
    except Exception as error:
        manifest.update(state='failed',error=repr(error));raise
    finally:
        manifest['finished_unix']=time.time();write(output/'gpt-运行记录.json',manifest)
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--potential-sign',type=int,choices=(-1,1),required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--construction-capacity-repair',action='store_true')
    args=parser.parse_args();run(args.output,args.potential_sign,args.resume,args.construction_capacity_repair)
