"""No-training replay/stress diagnostics for PhysX aggregate-pair overflow."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def child(output,layout,multiplier,steps,contact_pairs=2**23):
    from .pace_v2_formal import FormalEnv
    from .final_evaluation import load_policy
    import torch
    from dataclasses import replace
    from .config import STAGE1_CONFIG
    from .construction_staging import ConstructionStagingMixin
    libc=ctypes.CDLL(None)
    def event(phase,**values):
        libc.fflush(None)
        print(json.dumps(dict(event=phase,**values)),flush=True)
    event('before_construct',layout=layout,multiplier=multiplier)
    class Env(FormalEnv):
        def _create_envs(self):
            event('before_create_actors')
            super()._create_envs()
            event('after_create_actors')
        def _acquire_tensors(self):
            # Base __init__ calls gym.prepare_sim immediately before this hook.
            event('after_prepare_sim_before_acquire_tensors')
            super()._acquire_tensors()
            event('after_acquire_tensors')
        def _reset_idx(self,ids):
            initial=self.common_step_counter==0
            if initial:event('before_initial_reset')
            super()._reset_idx(ids)
            if initial:event('after_initial_reset')
        def _create_sim(self):
            original=self.gym
            class Proxy:
                def __getattr__(self,name):return getattr(original,name)
                def create_sim(proxy_self,device,graphics,engine,params):
                    params.physx.default_buffer_size_multiplier=multiplier
                    params.physx.max_gpu_contact_pairs=contact_pairs
                    event('sim_parameters',multiplier=params.physx.default_buffer_size_multiplier,
                          max_gpu_contact_pairs=params.physx.max_gpu_contact_pairs)
                    return original.create_sim(device,graphics,engine,params)
            self.gym=Proxy()
            try:return super()._create_sim()
            finally:self.gym=original
        def _make_origins(self):
            origins=super()._make_origins()
            if layout in ('concentrated','staged_concentrated'):
                self.terrain_levels[:]=9
                origins=self.terrain.env_origins[self.terrain_levels.cpu().numpy(),self.terrain_types.cpu().numpy()].copy()
            return origins
    class StagedEnv(ConstructionStagingMixin,Env):pass
    concentrated=layout in ('concentrated','staged_concentrated')
    cfg=STAGE1_CONFIG if not concentrated else replace(STAGE1_CONFIG,terrain=replace(STAGE1_CONFIG.terrain,curriculum=False))
    torch.set_num_threads(4)
    env=None
    try:
        env_class=StagedEnv if layout.startswith('staged') else Env
        env=env_class(num_envs=4096,seed=0,config=cfg)
        def occupancy():
            _,counts=torch.unique(env.env_origins,dim=0,return_counts=True)
            return dict(unique_origins=len(counts),maximum_robots_per_origin=int(counts.max()),
                        within_origin_unordered_pairs=int((counts*(counts-1)//2).sum()))
        event('constructed',**occupancy())
        torch.save(dict(root=env.root_states.cpu(),dof=env.dof_state.cpu(),obs=env.get_observations().cpu(),
                        critic=env.get_privileged_observations().cpu(),origins=env.env_origins.cpu(),
                        env_rng=env.generator.get_state().cpu(),torch_rng=torch.get_rng_state()),output/'gpt-初始状态.pt')
        model,norms=load_policy(ROOT/'artifacts/ppo/gpt-pace-v2-formal-stage2-seed0/gpt-model-30000.pt',env.device)
        initial={k:v.clone() for k,v in norms.state_dict().items()}
        env.set_training_iteration(29999)
        min_height=float('inf');max_height=-float('inf')
        for step in range(steps):
            event('before_step',step=step)
            if concentrated and step>0 and step%200==0:
                event('before_synchronized_reset',step=step)
                env.reset()
                event('after_synchronized_reset',step=step)
            env.reset_metrics()  # No retained raw-action history across the replay.
            with torch.no_grad():
                actions=model.act_inference(env.get_observations())
                _,_,reward,_,_=env.step(actions)
            if not torch.isfinite(reward).all() or not torch.isfinite(env.root_states).all():raise RuntimeError('non-finite replay state')
            min_height=min(min_height,float(env.root_states[:,2].min()))
            max_height=max(max_height,float(env.root_states[:,2].max()))
        event('after_replay',steps=steps,**occupancy())
        if not all(torch.equal(v,initial[k]) for k,v in norms.state_dict().items()):raise RuntimeError('normalizer changed')
        result=dict(layout=layout,multiplier=multiplier,contact_pairs=contact_pairs,steps=steps,num_envs=4096,seed=0,
                    classification='ENGINEERING_REPLAY_NO_PPO',normalization_frozen=True,
                    final_occupancy=occupancy(),root_world_height_min=min_height if steps else None,root_world_height_max=max_height if steps else None,
                    gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        (output/'gpt-子进程结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    finally:
        event('before_destroy')
        if env is not None:env.close()
        event('after_destroy')


def run(output,steps=2000,layout=None,multiplier=None,contact_pairs=2**23):
    output.mkdir(parents=True,exist_ok=True)
    variants=[('natural',5.),('concentrated',5.),('concentrated',20.)] if layout is None else [(layout,multiplier)]
    results=[]
    pattern=re.compile(r'will miss interactions|invalid parameter|CUDA error|out of memory|Fatal Python error',re.I)
    for layout,multiplier in variants:
        directory=output/('gpt-%s-buffer-%g'%(layout,multiplier));directory.mkdir(exist_ok=True)
        marker=directory/'gpt-审核结果.json'
        if marker.exists():results.append(json.loads(marker.read_text()));continue
        cmd=[sys.executable,'-m','pace_stage1.physics_capacity_audit','--output',str(directory),'--child','--layout',layout,'--multiplier',str(multiplier),'--steps',str(steps),'--contact-pairs',str(contact_pairs)]
        start=time.time();phase=None;warnings=[]
        with (directory/'gpt-带时间戳原生日志.log').open('w') as stream:
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                elapsed=time.time()-start
                stream.write('%.6f %s'%(elapsed,line));stream.flush()
                if line.startswith('{'):
                    try:phase=json.loads(line)
                    except json.JSONDecodeError:pass
                if pattern.search(line):warnings.append(dict(elapsed_seconds=elapsed,last_event=phase,message=line.strip()))
            code=process.wait()
        result=dict(layout=layout,multiplier=multiplier,contact_pairs=contact_pairs,steps=steps,returncode=code,
                    elapsed_seconds=time.time()-start,native_warnings=warnings,
                    runtime_clean=code==0 and not warnings and (directory/'gpt-子进程结果.json').exists())
        marker.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        (output/'gpt-诊断进度.json').write_text(json.dumps(dict(completed_variants=len(results),required_variants=len(variants),last_result=result),ensure_ascii=False,indent=2)+'\n')
    (output/'gpt-诊断汇总.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--child',action='store_true')
    p.add_argument('--layout',choices=('natural','concentrated','staged','staged_concentrated'));p.add_argument('--multiplier',type=float)
    p.add_argument('--contact-pairs',type=int,default=2**23)
    a=p.parse_args()
    if a.child:child(a.output,a.layout,a.multiplier,a.steps,a.contact_pairs)
    else:run(a.output,a.steps,a.layout,a.multiplier,a.contact_pairs)
