"""V0/V1 non-formal validation launcher with read-only protocol instrumentation."""
from .pace_v2_env import PaceV2ValidationEnv  # Isaac Gym before torch

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import torch

from .pace_v2_runner import ValidationRunner, validation_train_cfg
from .pace_v2_profiles import PaceV2PPOConfig
from .pace_v2_entropy import pace_v2_entropy_coefficient
from .semantics import foot_touchdown_schedule
from .protocol_gate import require_validation_training_allowed


class InstrumentationMixin:
    """Read-only protocol metrics shared by validation and formal environments."""
    def __init__(self,**kwargs):
        self.eq9_calls=0
        super().__init__(**kwargs)
        self.reset_metrics()
    def reset_metrics(self):
        self.audit_sums=torch.zeros(5,device=self.device)
        self.audit_substeps=0
        self.completed_lengths=[]
    def _step_actuator(self,target):
        q=self.dof_pos
        band=self.cfg.action.soft_limit_band_rad
        active=((q>self.upper_limits-band)&(target>self.upper_limits))|((q<self.lower_limits+band)&(target<self.lower_limits))
        result=super()._step_actuator(target)
        self.eq9_calls+=1
        self.audit_substeps+=1
        self.audit_sums+=torch.stack((active.float().mean(),((q<self.lower_limits)|(q>self.upper_limits)).float().mean(),(abs(result.raw_pd_torque-result.saturated_torque)>1e-6).float().mean(),result.raw_pd_torque.abs().mean(),result.applied_torque.abs().mean()))
        return result
    def step(self,actions):
        lengths=self.episode_length_buf.clone()+1
        result=super().step(actions)
        obs,critic,reward,done,infos=result
        terms=infos['reward_terms']
        k=foot_touchdown_schedule(self.training_iteration)
        expected=.01*(.2*terms['velocity_tracking']-terms['collision']-.1*k*terms['foot_touchdown'])
        if not torch.allclose(reward,expected,atol=1e-7,rtol=1e-5): raise RuntimeError('runtime reward differs from preregistration')
        if not torch.equal(terms['task_before_clip'],terms['task_after_clip']): raise RuntimeError('unexpected reward clipping')
        if done.any(): self.completed_lengths.extend(lengths[done].cpu().tolist())
        return result
    def metrics(self):
        names=('eq9_active_ratio','q_limit_ratio','torque_saturation_ratio','pd_torque_abs','applied_torque_abs')
        result=dict(zip(names,(self.audit_sums/max(1,self.audit_substeps)).cpu().tolist()))
        result['eq9_calls']=self.eq9_calls
        if self.completed_lengths: result['completed_episode_steps']=sum(self.completed_lengths)/len(self.completed_lengths)
        return result


class InstrumentedEnv(InstrumentationMixin, PaceV2ValidationEnv):
    pass


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(stage,output,terrain="plane"):
    require_validation_training_allowed()
    count,iterations,seed=(64,10,123) if stage=='v0' else (4096,300,0)
    output.mkdir(parents=True,exist_ok=False)
    cfg=validation_train_cfg(iterations)
    cfg["seed"]=seed
    manifest={'stage':stage,'classification':'SMOKE / NON-EXPERIMENTAL' if stage=='v0' else 'SHORT VALIDATION / NON-FORMAL','num_envs':count,'iterations':iterations,'seed':seed,'terrain':terrain,'formal':False,'ppo_profile':asdict(PaceV2PPOConfig()),'train_cfg':cfg,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(0),'state':'constructing','started_unix':time.time()}
    write(output/'gpt-运行记录.json',manifest)
    env=None
    try:
        torch.set_num_threads(4)
        env=InstrumentedEnv(num_envs=count,max_iterations=iterations,seed=seed,terrain_mode=terrain,sim_device='cuda:0',headless=True)
        runner=ValidationRunner(env,cfg,log_dir=str(output),device='cuda:0')
        assert runner.alg.num_mini_batches==10 and runner.num_steps_per_env==24
        assert runner.alg.actor_critic.actor[0] is runner.normalizers.actor
        assert runner.alg.actor_critic.critic[0] is runner.normalizers.critic
        manifest['state']='running'
        write(output/'gpt-运行记录.json',manifest)
        with (output/'gpt-逐迭代指标.jsonl').open('w') as stream:
            for i in range(iterations):
                env.reset_metrics()
                row=runner.iteration()
                row.update(env.metrics())
                expected_count=count*(1+24*(i+1))
                if row['actor_count']!=expected_count or row['critic_count']!=expected_count: raise RuntimeError('normalizer sample count mismatch')
                if row['entropy_coef']!=pace_v2_entropy_coefficient(i,PaceV2PPOConfig().entropy): raise RuntimeError('entropy injection mismatch')
                if env.eq9_calls!=(i+1)*24*4: raise RuntimeError('Eq9 substep wiring mismatch')
                if not all(torch.isfinite(x).all() for x in runner.alg.actor_critic.state_dict().values()): raise RuntimeError('non-finite model/normalizer state')
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(json.dumps(row,allow_nan=False),flush=True)
                if (i+1)%50==0: runner.save(output/('gpt-validation-%d.pt'%(i+1)))
        checkpoint=output/'gpt-validation-final.pt'
        runner.save(checkpoint)
        # Preserve identical observations; restore against a newly constructed environment.
        obs=env.get_observations().clone();critic=env.get_privileged_observations().clone()
        def probe(r):
            r.alg.actor_critic.eval()
            with torch.no_grad(): return [r.normalizers.actor(obs),r.normalizers.critic(critic),r.alg.actor_critic.act_inference(obs),r.alg.actor_critic.evaluate(critic)]
        expected=probe(runner)
        env.close()
        env=InstrumentedEnv(num_envs=64,max_iterations=iterations,seed=seed,terrain_mode=terrain,sim_device='cuda:0',headless=True)
        fresh=ValidationRunner(env,cfg,log_dir=str(output),device='cuda:0')
        fresh.load(checkpoint)
        if not all(torch.equal(a,b) for a,b in zip(expected,probe(fresh))): raise RuntimeError('GPU checkpoint closure mismatch')
        if fresh.current_learning_iteration!=iterations: raise RuntimeError('checkpoint next-update mismatch')
        manifest.update(state='completed',protocol_checks='PASS',checkpoint_closure='bitwise_equal',next_iteration=fresh.current_learning_iteration,finished_unix=time.time())
    except Exception as error:
        manifest.update(state='failed',error=repr(error),finished_unix=time.time())
        raise
    finally:
        write(output/'gpt-运行记录.json',manifest)
        if env is not None: env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('v0','v1'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--terrain', choices=('plane','trimesh'), default='plane')
    args=parser.parse_args()
    run(args.stage,args.output,args.terrain)
