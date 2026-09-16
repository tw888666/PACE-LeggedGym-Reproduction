"""Descriptive fixed-case replay; no optimizer, reward change or model selection."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv'


def foot_sphere_offsets(names):
    """URDF fixed-chain transforms from retained shank frame to foot sphere."""
    import numpy as np
    import xml.etree.ElementTree as ET
    root=ET.parse(ROOT/'third_party/anymal_d_simple_description/urdf/anymal.urdf').getroot()
    joints={j.find('child').attrib['link']:j for j in root.findall('joint')}
    def transform(origin):
        xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
        r,p,y=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')
        cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
        rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        out=np.eye(4);out[:3,:3]=rotation;out[:3,3]=xyz
        return out
    offsets=[]
    for name in names:
        foot=name.split('_')[0]+'_FOOT'
        link=root.find("link[@name='%s']"%foot)
        sphere=next(c for c in link.findall('collision') if c.find('geometry/sphere') is not None)
        matrix=transform(sphere.find('origin'))
        node=foot
        while node!=name:
            j=joints[node]
            if j.attrib['type']!='fixed':raise ValueError('foot transform crosses movable joint')
            matrix=transform(j.find('origin'))@matrix
            node=j.find('parent').attrib['link']
        offsets.append(matrix[:3,3])
    return np.stack(offsets)


def selected_cases():
    with PLAN.open() as stream:
        return [c for c in csv.DictReader(stream)
                if c['seed'] == '200000' and c['replicate'] == '0'
                and c['command_bin'] == 'normal_motion'
                and (c['difficulty_row'] == '5' or c['terrain'] == 'flat_reference')]


def summarize(arrays, dt=.01):
    import numpy as np
    contact = arrays['contact'].astype(bool)
    foot_xy = np.linalg.norm(arrays['foot_velocity'][:, :, :2], axis=-1)
    legs = []
    for j in range(4):
        c = contact[:, j]
        touchdowns = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        periods = np.diff(touchdowns) * dt
        # Complete stance/swing runs only; truncate neither endpoint into a cycle.
        edges = np.flatnonzero(c[1:] != c[:-1]) + 1
        lengths = np.diff(edges) * dt
        states = c[edges[:-1]]
        mean = lambda x: float(np.mean(x)) if len(x) else None
        legs.append(dict(duty_factor=float(c.mean()),
                         touchdown_events=int(len(touchdowns)),
                         touchdown_event_hz=float(len(touchdowns)/(len(c)*dt)),
                         median_cycle_s=float(np.median(periods)) if len(periods) else None,
                         cycle_frequency_hz=float(1/np.median(periods)) if len(periods) else None,
                         complete_stance_mean_s=mean(lengths[states]),
                         complete_swing_mean_s=mean(lengths[~states]),
                         contact_horizontal_speed_mean_m_s=mean(foot_xy[c, j]),
                         contact_horizontal_speed_rms_m_s=float(np.sqrt(np.mean(foot_xy[c,j]**2))) if c.any() else None))
    return dict(duration_s=len(contact)*dt,
                joint_velocity_rms_rad_s=np.sqrt(arrays['qdot_squared_substep_mean'].mean(axis=0)).tolist(),
                joint_velocity_rms_all_rad_s=float(np.sqrt(arrays['qdot_squared_substep_mean'].mean())),
                clipped_offset_abs_mean_rad=float(np.abs(.5*np.clip(arrays['raw_action'],-100,100)).mean()),
                raw_action_abs_mean=float(np.abs(arrays['raw_action']).mean()),
                action_rate_rms_raw_per_s=float(np.sqrt(np.mean((np.diff(arrays['raw_action'],axis=0)/dt)**2))),
                torque_saturation=float(arrays['saturation'].mean()),
                legs=legs)


def replay(case, directory, checkpoint):
    # Native simulator must load before torch.
    from .final_evaluation_env import FinalEvaluationEnv
    from .final_evaluation import load_policy
    import numpy as np
    import torch
    from isaacgym.torch_utils import quat_rotate
    torch.set_num_threads(4)

    class AuditEnv(FinalEvaluationEnv):
        def _step_actuator(self, target):
            self.audit_qdot2 += self.dof_vel.square()/self.cfg.action.policy_decimation
            return super()._step_actuator(target)

    env = AuditEnv([case], directory.parent/'gpt-地形缓存')
    try:
        model, norms = load_policy(checkpoint, env.device)
        original = {k: v.clone() for k,v in norms.state_dict().items()}
        names=[env.body_names[int(i)] for i in env.foot_indices.cpu().tolist()]
        offsets=torch.tensor(foot_sphere_offsets(names),device=env.device,dtype=torch.float32)
        history = {k: [] for k in ('qdot_squared_substep_mean','contact','foot_position','shank_velocity','raw_action','saturation','q','qdot','root_state')}
        with torch.no_grad():
            for step in range(2001):
                action = model.act_inference(env.get_observations())
                env.audit_qdot2 = torch.zeros_like(env.dof_vel)
                _,_,_,done,_ = env.step(action)
                values = dict(qdot_squared_substep_mean=env.audit_qdot2[0],
                              contact=env.contact_forces[0,env.foot_indices,2]>env.cfg.rewards.foot_contact_threshold_n,
                              foot_position=env.rigid_body_state[0,env.foot_indices,:3]+quat_rotate(env.rigid_body_state[0,env.foot_indices,3:7],offsets),
                              shank_velocity=env.rigid_body_state[0,env.foot_indices,7:10],
                              raw_action=action[0], saturation=env.per_joint[0],
                              q=env.dof_pos[0], qdot=env.dof_vel[0],root_state=env.root_states[0])
                for k,v in values.items():
                    if not torch.isfinite(v).all(): raise RuntimeError('non-finite gait state')
                    history[k].append(v.cpu().numpy().copy())
                if (step+1)%500==0:print('policy steps',step+1,flush=True)
                if done[0]: break
        if not all(torch.equal(v, original[k]) for k,v in norms.state_dict().items()):
            raise RuntimeError('normalizer updated during replay')
        arrays = {k: np.stack(v) for k,v in history.items()}
        arrays['foot_velocity']=np.gradient(arrays['foot_position'],.01,axis=0)
        np.savez_compressed(directory/'gpt-步态时序.npz', **arrays)
        result = dict(case=case, checkpoint=str(checkpoint), classification='POST_HOC_DESCRIPTIVE_REPLAY',
                      normalization_frozen=True, metrics=summarize(arrays),
                      foot_body_names=[env.body_names[int(i)] for i in env.foot_indices.cpu().tolist()] if hasattr(env,'body_names') else None)
        (directory/'gpt-步态结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    finally:
        env.close()


def run(output, checkpoint, case_id=None, audit_case=None):
    cases = selected_cases()
    if audit_case is not None:
        cases=[c for c in cases if int(c['case_id'])==audit_case]
        if not cases:raise ValueError('case is outside fixed audit selection')
    output.mkdir(parents=True, exist_ok=True)
    if case_id is not None:
        case = next(c for c in cases if int(c['case_id']) == case_id)
        directory = output/('gpt-case-%s'%case_id)
        directory.mkdir(exist_ok=True)
        replay(case,directory,checkpoint)
        return
    for c in cases:
        directory=output/('gpt-case-%s'%c['case_id']);directory.mkdir(exist_ok=True)
        with (directory/'gpt-调度锁').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=directory/'gpt-步态结果.json'
            if path.exists() and json.loads(path.read_text()).get('runtime_log_checked'):
                if Path(json.loads(path.read_text())['checkpoint']).resolve()!=checkpoint.resolve():raise RuntimeError('different checkpoint in output')
                continue
            with (directory/'gpt-原生运行日志.log').open('w') as stream:
                child=subprocess.run([sys.executable,'-m','pace_stage1.gait_audit','--output',str(output),'--checkpoint',str(checkpoint),'--case-id',c['case_id']],stdout=stream,stderr=subprocess.STDOUT)
            log=(directory/'gpt-原生运行日志.log').read_text(errors='replace')
            if child.returncode or re.search(r'will miss interactions|invalid parameter|Fatal Python error|CUDA error|out of memory',log,re.I) or not path.exists():
                raise RuntimeError('gait replay failed: '+str(directory))
            data=json.loads(path.read_text());data['runtime_log_checked']=True
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
            print('completed',c['case_id'],c['terrain'],flush=True)
    if audit_case is None:
        (output/'gpt-完成记录.json').write_text(json.dumps(dict(state='completed',case_ids=[c['case_id'] for c in cases],checkpoint=str(checkpoint)),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/ppo/gpt-pace-v2-formal-stage1-seed0/gpt-model-30000.pt')
    parser.add_argument('--case-id',type=int)
    parser.add_argument('--audit-case',type=int,help='Run and audit one of the six fixed cases')
    args=parser.parse_args();run(args.output,args.checkpoint,args.case_id,args.audit_case)
