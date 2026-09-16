"""Run the preregistered final case table, without policy updates or checkpoint selection."""
from .final_evaluation_env import FinalEvaluationEnv, gymapi
import argparse,csv,json,time,os,subprocess,fcntl,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from .empirical_normalization import ActorCriticEmpiricalNormalizers, OBSERVATION_NORMALIZATION_SPEC
from .semantics import quat_rotate_inverse

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv'
DEFAULT_CHECKPOINT=ROOT/'artifacts/ppo/gpt-pace-v2-formal-stage1-seed0/gpt-model-30000.pt'


class VideoWriter:
    def __init__(self,path):
        self.log=path.with_suffix('.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','640x480','-framerate','25','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','20','-threads','2',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
    def append_data(self,frame):self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
    def close(self):
        self.process.stdin.close()
        code=self.process.wait(timeout=60);self.log.close()
        if code:raise RuntimeError('ffmpeg failed with code %d'%code)


def load_policy(checkpoint,device):
    data=torch.load(checkpoint,map_location=device)
    if data['iter']!=30000:raise RuntimeError('final evaluation requires iteration 30000')
    if data['observation_normalization']['spec']!=OBSERVATION_NORMALIZATION_SPEC:raise RuntimeError('normalization spec mismatch')
    model=ActorCritic(48,353,12,actor_hidden_dims=[256,256,256,128],critic_hidden_dims=[256,256,256,128],activation='elu',init_noise_std=1.5).to(device)
    norms=ActorCriticEmpiricalNormalizers(48,353).to(device)
    model.actor=nn.Sequential(norms.actor,model.actor);model.critic=nn.Sequential(norms.critic,model.critic)
    model.load_state_dict(data['model_state_dict'],strict=True);model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise RuntimeError('non-finite checkpoint')
    return model,norms


def write_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n');os.replace(temp,path)


def evaluate_group(cases,directory,checkpoint,video=False):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'gpt-执行锁').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        _evaluate_group(cases,directory,checkpoint,video)
        return True


def _evaluate_group(cases,directory,checkpoint,video=False):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'gpt-完成记录.json').exists():return
    started=time.time();env=None;writers={};cameras={}
    try:
        torch.set_num_threads(4)
        env=FinalEvaluationEnv(cases,directory.parent/'gpt-地形缓存',video=video)
        model,norms=load_policy(checkpoint,env.device)
        initial_norm={k:v.clone() for k,v in norms.state_dict().items()}
        n=len(cases);dev=env.device
        lengths=torch.zeros(n,device=dev);errors=torch.zeros((n,2),device=dev)
        control=torch.zeros((n,6),device=dev);joint=torch.zeros((n,12),device=dev)
        action_sum=torch.zeros(n,device=dev);tail=torch.zeros((n,3),device=dev)
        raw_min=torch.full((n,12),float('inf'),device=dev);raw_max=-raw_min
        safe_min=raw_min.clone();safe_max=raw_max.clone()
        survived=torch.zeros(n,dtype=torch.bool,device=dev);crossed=torch.zeros_like(survived)
        history=torch.full((2001,n,12),float('nan'),device=dev)
        if video:
            for i,c in enumerate(cases):
                props=gymapi.CameraProperties();props.width=640;props.height=480
                camera=env.gym.create_camera_sensor(env.envs[i],props)
                if camera<0:raise RuntimeError('camera creation failed')
                cameras[i]=camera
                writers[i]=VideoWriter(directory/('gpt-case-%s.mp4'%c['case_id']))
        with torch.no_grad():
            for step in range(2001):
                active=env.active.clone()
                actions=model.act_inference(env.get_observations())
                if not torch.isfinite(actions[active]).all():raise RuntimeError('non-finite active action')
                actions[~active]=0
                history[step,active]=actions[active]
                _,_,_,done,_=env.step(actions)
                q=env.root_states[:,3:7]
                linear=quat_rotate_inverse(q,env.root_states[:,7:10]);angular=quat_rotate_inverse(q,env.root_states[:,10:13])
                squared=torch.stack(((linear[:,:2]-env.commands[:,:2]).square().sum(1),(angular[:,2]-env.commands[:,2]).square()),1)
                if not torch.isfinite(squared[active]).all():raise RuntimeError('non-finite active state')
                lengths+=active;errors+=torch.where(active[:,None],squared,0)
                control+=torch.where(active[:,None],env.control,0);joint+=torch.where(active[:,None],env.per_joint,0)
                action_sum+=torch.where(active,actions.abs().mean(1),0)
                tail+=torch.stack([((actions.abs()>v).float().mean(1))*active for v in (2,5,10)],1)
                raw_min=torch.where(active[:,None],torch.minimum(raw_min,env.raw_min),raw_min)
                raw_max=torch.where(active[:,None],torch.maximum(raw_max,env.raw_max),raw_max)
                safe_min=torch.where(active[:,None],torch.minimum(safe_min,env.safe_min),safe_min)
                safe_max=torch.where(active[:,None],torch.maximum(safe_max,env.safe_max),safe_max)
                displacement=(env.root_states[:,:2]-env.env_origins[:,:2]).abs()
                crossed|=active&(displacement.amax(1)>4.)
                if env.terrain_mode!='plane' and (active&(displacement.amax(1)>34.)).any():raise RuntimeError('active case exceeded tiled mesh safety extent')
                contact=(env.contact_forces[:,env.base_indices,:].norm(dim=2)>env.cfg.rewards.termination_contact_threshold_n).any(1)
                survived|=active&done.bool()&env.time_out_buf&~contact
                if video and step%4==0:
                    from PIL import Image,ImageDraw
                    env.gym.step_graphics(env.sim)
                    for i,camera in cameras.items():
                        if not active[i]:continue
                        pos=env.root_states[i,:3].cpu().tolist()
                        env.gym.set_camera_location(camera,env.envs[i],gymapi.Vec3(pos[0]+2.8,pos[1]-2.8,pos[2]+1.7),gymapi.Vec3(*pos))
                    env.gym.render_all_camera_sensors(env.sim)
                    for i,camera in cameras.items():
                        if not active[i]:continue
                        rgba=env.gym.get_camera_image(env.sim,env.envs[i],camera,gymapi.IMAGE_COLOR).reshape(480,640,4)
                        frame=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(frame)
                        draw.rectangle((0,0,640,38),fill=(0,0,0))
                        draw.text((8,5),'case %s | %s | row %s | %s | %.2fs'%(cases[i]['case_id'],cases[i]['terrain'],cases[i]['difficulty_row'],cases[i]['command_bin'],(step+1)*.01),fill=(255,255,255))
                        writers[i].append_data(np.asarray(frame))
                env.active &= ~done.bool()
                env.park_inactive()
                if (step+1)%200==0:
                    write_json(directory/'gpt-执行进度.json',dict(state='running',policy_steps=step+1,
                        remaining_active_cases=int(env.active.sum()),elapsed_seconds=time.time()-started,video=video))
                if not env.active.any():break
        if env.active.any():raise RuntimeError('planned horizon reached without terminal classification')
        if not all(torch.equal(v,initial_norm[k]) for k,v in norms.state_dict().items()):raise RuntimeError('normalizer changed in final evaluation')
        raw=history.cpu().numpy();counts=lengths.cpu().numpy().astype(int)
        if not video:np.savez_compressed(directory/'gpt-原始动作样本.npz',actions=raw,lengths=counts,case_ids=np.array([int(c['case_id']) for c in cases]))
        rmse=(errors/lengths[:,None]).sqrt().cpu().numpy();means=(control/lengths[:,None]).cpu().numpy()
        records=[]
        for i,c in enumerate(cases):
            r=dict(c);r.update(steps=int(counts[i]),duration_s=float(counts[i])*.01,velocity_rmse=float(rmse[i,0]),yaw_rmse=float(rmse[i,1]),survival=bool(survived[i]),joint_task_success=bool(survived[i] and rmse[i,0]<=.3 and rmse[i,1]<=.3),mean_abs_mu=float(action_sum[i]/lengths[i]),crossed_tile=bool(crossed[i]))
            for j,k in enumerate(('saturation','eq9_activation','hard_limit_exceedance','electrical_w','mechanical_signed_w','mechanical_positive_net_w')):r[k]=float(means[i,j])
            r['diagnostic_power_w']=r['electrical_w']+r['mechanical_positive_net_w']
            for k in ('electrical','mechanical_signed','mechanical_positive_net','diagnostic_power'):r[k+'_j']=r[k+'_w']*r['duration_s']
            r['per_joint_saturation']=(joint[i]/lengths[i]).cpu().tolist()
            for name,value in [('raw_target_min',raw_min),('raw_target_max',raw_max),('safe_target_min',safe_min),('safe_target_max',safe_max)]:r[name]=value[i].cpu().tolist()
            for j,t in enumerate((2,5,10)):r['prob_abs_mu_gt_%d'%t]=float(tail[i,j]/lengths[i])
            records.append(r)
        write_json(directory/'gpt-逐回合指标.json',records)
        for writer in writers.values():writer.close()
        writers.clear()
        write_json(directory/'gpt-完成记录.json',dict(state='completed',case_count=n,checkpoint=str(checkpoint),video=video,normalization_frozen=True,checkpoint_std=model.std.cpu().tolist(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(0),seconds=time.time()-started,finished_unix=time.time()))
    except Exception as e:
        write_json(directory/'gpt-错误记录.json',dict(error=repr(e),state='failed'));raise
    finally:
        for writer in writers.values():writer.close()
        if env is not None and not video:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);parser.add_argument('--video',action='store_true');parser.add_argument('--group',type=int)
    parser.add_argument('--work-queue',action='store_true',help='Claim unfinished groups using per-group locks')
    args=parser.parse_args();cases=list(csv.DictReader(PLAN.open()))
    keys=list(dict.fromkeys((c['terrain'],c['difficulty_row']) for c in cases))
    for index,key in enumerate(keys):
        if (not args.work_queue and index%args.shards!=args.shard) or (args.group is not None and index!=args.group):continue
        selected=[c for c in cases if (c['terrain'],c['difficulty_row'])==key]
        if args.video:
            selected=[c for c in selected if c['seed']=='200000' and c['replicate']=='0' and (c['difficulty_row'] in ('2','5','8') or c['terrain']=='flat_reference')]
            if not selected:continue
        name='gpt-%02d-%s-%s'%(index,*key)
        if args.video and args.group is None:
            if (args.output/name/'gpt-完成记录.json').exists():
                continue
            # Preview 4 can crash when recreating a graphics simulator in one process.
            # Isolate graphics lifetimes; each child retains the same fixed cases.
            subprocess.run([sys.executable,'-m','pace_stage1.final_evaluation',
                            '--output',str(args.output),'--checkpoint',str(args.checkpoint),
                            '--video','--group',str(index),'--work-queue'],check=True)
            continue
        if evaluate_group(selected,args.output/name,args.checkpoint,args.video):
            print('completed group',index,key,flush=True)

if __name__=='__main__':
    main()
    if '--video' in sys.argv and '--group' in sys.argv:
        # Graphics workers are process-isolated. Preview 4 may crash in graphics
        # teardown after all frames and metrics have been flushed successfully.
        # Release the native context with process exit, preserving all output.
        import ctypes
        torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os._exit(0)
