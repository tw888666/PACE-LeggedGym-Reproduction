"""Signed PACE-cost readout on the unchanged fixed final-evaluation cases."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv'


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def groups():
    with PLAN.open() as stream:cases=list(csv.DictReader(stream))
    keys=list(dict.fromkeys((c['terrain'],c['difficulty_row']) for c in cases))
    return [[c for c in cases if (c['terrain'],c['difficulty_row'])==key] for key in keys]


def replay(cases,directory,checkpoint):
    from .final_evaluation_env import FinalEvaluationEnv
    from .final_evaluation import load_policy
    from .stage2_energy import energy_components
    from .stage3_cost import EnergyCostSpec
    import torch
    class CostEnv(FinalEvaluationEnv):
        def _step_actuator(self,target):
            result=super()._step_actuator(target)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.cost_components+=energy_components(result.applied_torque,self.dof_vel,
                self.body_mass,self.rigid_body_state[:,:,9],potential_sign=-1)/4
            return result
    started=time.time();torch.set_num_threads(4)
    env=CostEnv(cases,directory.parent/'gpt-地形缓存')
    try:
        model,norms=load_policy(checkpoint,env.device)
        original={k:v.clone() for k,v in norms.state_dict().items()}
        spec=EnergyCostSpec('pace_corrected',env.policy_dt)
        count=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        totals=torch.zeros((env.num_envs,6),device=env.device,dtype=torch.float64)
        negative=torch.zeros_like(count)
        with torch.no_grad():
            for step in range(2001):
                active=env.active.clone()
                action=model.act_inference(env.get_observations())
                if not torch.isfinite(action[active]).all():raise RuntimeError('non-finite action')
                action[~active]=0
                env.cost_components=torch.zeros((env.num_envs,3),device=env.device)
                _,_,_,done,_=env.step(action)
                power=spec.power_w(env.cost_components,env.commands)
                if not torch.isfinite(power[active]).all():raise RuntimeError('non-finite active cost')
                # Signed, positive and negative contributions are all retained.
                values=torch.cat((env.cost_components,power[:,None],power.clamp_min(0)[:,None],power.clamp_max(0)[:,None]),1)
                totals+=torch.where(active[:,None],values.to(torch.float64),0.)*env.policy_dt
                count+=active;negative+=active&(power<0)
                env.active&=~done.bool();env.park_inactive()
                if (step+1)%200==0:
                    write(directory/'gpt-执行进度.json',dict(policy_steps=step+1,active_cases=int(env.active.sum()),seconds=time.time()-started))
                if not env.active.any():break
        if env.active.any():raise RuntimeError('cases still active beyond the fixed horizon')
        if not all(torch.equal(v,original[k]) for k,v in norms.state_dict().items()):raise RuntimeError('normalizer updated')
        records=[]
        for i,case in enumerate(cases):
            seconds=int(count[i])*env.policy_dt
            names=('electrical_j','mechanical_positive_net_j','potential_j','corrected_cost_j','positive_corrected_j','negative_corrected_j')
            row=dict(case,steps=int(count[i]),duration_s=seconds,negative_step_count=int(negative[i]))
            row.update(zip(names,totals[i].cpu().tolist()))
            row['corrected_mean_w']=row['corrected_cost_j']/seconds
            records.append(row)
        write(directory/'gpt-逐回合完整功率.json',records)
        write(directory/'gpt-完成记录.json',dict(state='completed',case_count=len(cases),checkpoint=str(checkpoint.resolve()),
             cost_definition='pace_corrected',potential_sign=-1,normalization_frozen=True,seconds=time.time()-started,
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    finally:env.close()


def run(output,checkpoint,worker_index=0,workers=1,group=None):
    selected=groups();output.mkdir(parents=True,exist_ok=True)
    if group is not None:
        directory=output/('gpt-group-%02d'%group);directory.mkdir(exist_ok=True)
        replay(selected[group],directory,checkpoint);return
    for i,cases in enumerate(selected):
        if i%workers!=worker_index:continue
        directory=output/('gpt-group-%02d'%i);directory.mkdir(exist_ok=True)
        with (directory/'gpt-调度锁').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            marker=directory/'gpt-完成记录.json'
            if marker.exists():
                m=json.loads(marker.read_text())
                if m['checkpoint']!=str(checkpoint.resolve()):raise RuntimeError('checkpoint mismatch')
                if m.get('runtime_log_checked'):continue
            log=directory/'gpt-原生运行日志.log'
            if log.exists():log.rename(directory/('gpt-历史日志-%d.log'%time.time_ns()))
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','pace_stage1.stage3_cost_calibration',
                    '--output',str(output),'--checkpoint',str(checkpoint),'--group',str(i)],stdout=stream,stderr=subprocess.STDOUT)
            invalid=re.search(r'will miss interactions|invalid parameter|CUDA error|out of memory|Fatal Python error',log.read_text(errors='replace'),re.I)
            if child.returncode or invalid or not marker.exists():
                write(directory/'gpt-审核失败.json',dict(returncode=child.returncode,native_error=invalid.group(0) if invalid else None))
                raise RuntimeError('cost calibration group failed: '+str(directory))
            m=json.loads(marker.read_text());m['runtime_log_checked']=True;write(marker,m)
            print('audited',i,len(cases),flush=True)


def report(output,checkpoint):
    records=[];planned={c['case_id']:c for batch in groups() for c in batch}
    markers=list(output.glob('gpt-group-*/gpt-完成记录.json'))
    if len(markers)!=51:raise RuntimeError('all 51 groups required')
    for marker in markers:
        m=json.loads(marker.read_text())
        if not m.get('runtime_log_checked') or not m['normalization_frozen'] or m['checkpoint']!=str(checkpoint.resolve()):raise RuntimeError('unaudited or mixed calibration')
        records+=json.loads((marker.parent/'gpt-逐回合完整功率.json').read_text())
    if len(records)!=29376 or len({r['case_id'] for r in records})!=29376:raise RuntimeError('case coverage mismatch')
    for r in records:
        if any(r[k]!=v for k,v in planned[r['case_id']].items()):raise RuntimeError('case plan mismatch')
    def aggregate(rows):
        energy=sum(r['corrected_cost_j'] for r in rows);duration=sum(r['duration_s'] for r in rows)
        return dict(case_count=len(rows),corrected_energy_j=energy,duration_s=duration,
                    time_weighted_mean_w=energy/duration,case_weighted_mean_w=sum(r['corrected_mean_w'] for r in rows)/len(rows),
                    negative_step_fraction=sum(r['negative_step_count'] for r in rows)/sum(r['steps'] for r in rows),
                    positive_corrected_j=sum(r['positive_corrected_j'] for r in rows),negative_corrected_j=sum(r['negative_corrected_j'] for r in rows))
    primary=[r for r in records if r['terrain']!='flat_reference']
    summary=dict(classification='FIXED_EVALUATION_DISTRIBUTION_COST_CALIBRATION',cost_definition='pace_corrected',
                 checkpoint=str(checkpoint.resolve()),primary=aggregate(primary),
                 by_terrain={t:aggregate([r for r in records if r['terrain']==t]) for t in sorted({r['terrain'] for r in records})},
                 formal_budget_selected=False,online_training_distribution_equivalence_claimed=False)
    entry_path=output/'gpt-标定入口.json'
    if entry_path.exists():
        reference=Path(json.loads(entry_path.read_text())['reference_evaluation'])
        original={}
        for p in (reference/'quantitative').glob('*/gpt-逐回合指标.json'):
            original.update({r['case_id']:r for r in json.loads(p.read_text())})
        differences=[abs((r['electrical_j']+r['mechanical_positive_net_j'])/r['duration_s']-original[r['case_id']]['diagnostic_power_w']) for r in records]
        summary['reference_replay_comparison']=dict(reference_evaluation=str(reference),
            duration_mismatch_cases=sum(r['steps']!=original[r['case_id']]['steps'] for r in records),
            mean_absolute_diagnostic_difference_w=sum(differences)/len(differences),
            maximum_absolute_diagnostic_difference_w=max(differences),
            bitwise_trajectory_equivalence_claimed=False)
    write(output/'gpt-完整功率标定汇总.json',summary)
    with (output/'gpt-逐回合完整功率.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--group',type=int);p.add_argument('--report',action='store_true')
    a=p.parse_args()
    if a.report:report(a.output,a.checkpoint)
    else:
        if not 0<=a.worker_index<a.workers:p.error('worker index outside worker count')
        run(a.output,a.checkpoint,a.worker_index,a.workers,a.group)
