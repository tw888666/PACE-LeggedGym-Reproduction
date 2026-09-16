"""Evaluation-only first-episode environment implementing FINAL_EVALUATION_V1."""
from .env import Stage1LocomotionEnv
from isaacgym import gymapi, gymtorch, terrain_utils
import os
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from .config import STAGE1_CONFIG
from .terrain import build_terrain
from .pace_v2_control import pace_v2_hard_limit_safe_target
from .pace_v2_rewards import pace_v2_environment_config
from .semantics import quat_rotate_inverse


def case_random(case, stream, shape):
    return np.random.default_rng(np.random.SeedSequence([int(case['seed']),int(case['case_id']),stream])).random(shape,dtype=np.float32)


def source_tile(seed,row,col,cache):
    path=Path(cache)/('gpt-source-terrain-%d.npy'%seed)
    if not path.exists():
        # Extract the unchanged generator's height map; mesh conversion is unnecessary here.
        original=terrain_utils.convert_heightfield_to_trimesh
        rng=np.random.get_state()
        try:
            np.random.seed(seed)
            terrain_utils.convert_heightfield_to_trimesh=lambda *args:(None,None)
            raw=build_terrain(STAGE1_CONFIG.terrain).height_field_raw
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.stem+"-%d.tmp.npy"%os.getpid())
            np.save(temporary,raw)
            os.replace(temporary,path)
        finally:
            terrain_utils.convert_heightfield_to_trimesh=original
            np.random.set_state(rng)
    raw=np.load(path,mmap_mode='r')
    return raw[250+row*80:250+(row+1)*80,250+col*80:250+(col+1)*80].copy()


class FinalEvaluationEnv(Stage1LocomotionEnv):
    def __init__(self,cases,cache,video=False):
        self.cases=cases;self.cache=cache;self.initializing=True
        self.fixed_commands=np.array([[float(c[k]) for k in ('vx_m_s','vy_m_s','yaw_rad_s')]+[0.] for c in cases],dtype=np.float32)
        self.noise_numpy=np.stack([case_random(c,1,(2002,48)) for c in cases],axis=1)
        self.video=video
        super().__init__(config=pace_v2_environment_config(STAGE1_CONFIG),num_envs=len(cases),sim_device='cuda:0',headless=not video,terrain_mode='plane' if cases[0]['terrain']=='flat_reference' else 'trimesh',seed=int(cases[0]['seed']))
        self.initializing=False
        self.active=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.initial_root=self.root_states.clone();self.initial_dof=self.dof_state.clone()
        self.set_training_iteration(29999)

    def _create_terrain(self):
        self.patch_origins=[];self.patch_indices=[]
        if self.terrain_mode=='plane':
            super()._create_terrain();return
        keys=sorted({(int(c['seed']),int(c['source_column'])) for c in self.cases})
        self.tiles=[]
        self.case_origins=np.zeros((len(self.cases),3),dtype=np.float32)
        cursor=0.
        for i,(seed,col) in enumerate(keys):
            tile=source_tile(seed,int(self.cases[0]['difficulty_row']),col,self.cache)
            self.tiles.append(tile)
            indices=[j for j,c in enumerate(self.cases) if (int(c['seed']),int(c['source_column']))==(seed,col)]
            side=int(np.ceil(np.sqrt(len(indices))))
            # Separate physical actors even though collision groups are filtered.
            # Coincident origins exhaust PhysX broadphase candidate buffers.
            # Whole-tile translations retain exactly the same local height map.
            tiled=np.tile(tile,(side+10,side+10))
            vertices,triangles=terrain_utils.convert_heightfield_to_trimesh(tiled,.1,.005,.75)
            p=gymapi.TriangleMeshParams();p.nb_vertices=len(vertices);p.nb_triangles=len(triangles)
            p.transform.p=gymapi.Vec3(cursor,0.,0.)
            p.static_friction=1.;p.dynamic_friction=1.;p.restitution=0.
            self.gym.add_triangle_mesh(self.sim,vertices.flatten(),triangles.flatten(),p)
            height=float(tile[30:50,30:50].max())*.005
            self.patch_origins.append([cursor+44.,44.,height])
            for offset,j in enumerate(indices):
                self.case_origins[j]=[cursor+44.+8.*(offset//side),44.+8.*(offset%side),height]
            cursor+=(side+10)*8.+16.
        self.patch_indices=[keys.index((int(c['seed']),int(c['source_column']))) for c in self.cases]
        self.terrain=SimpleNamespace(env_origins=np.asarray(self.patch_origins,dtype=np.float32))  # Homogeneous patches; curriculum disabled.

    def _make_origins(self):
        self.terrain_levels=torch.tensor([int(c['difficulty_row'] or 0) for c in self.cases],device=self.device)
        self.terrain_types=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        if self.terrain_mode=='plane':
            side=int(np.ceil(np.sqrt(self.num_envs)))
            return np.array([[8.*(i//side),8.*(i%side),0.] for i in range(self.num_envs)],dtype=np.float32)
        return self.case_origins.copy()

    def _create_envs(self):
        super()._create_envs()
        friction=np.array([.5+.75*case_random(c,2,(1,))[0] for c in self.cases])
        self.friction_coefficients=torch.tensor(friction,dtype=torch.float32,device=self.device)
        for i in range(self.num_envs):
            shapes=self.gym.get_actor_rigid_shape_properties(self.envs[i],self.actors[i])
            for shape in shapes:shape.friction=float(friction[i])
            self.gym.set_actor_rigid_shape_properties(self.envs[i],self.actors[i],shapes)

    def _resample_commands(self,ids):
        self.commands[:]=torch.as_tensor(self.fixed_commands,device=self.device)
    def _update_heading_commands(self):pass
    def _update_terrain_curriculum(self,ids):pass

    def _reset_idx(self,ids):
        if not self.initializing:return
        super()._reset_idx(ids)
        q=torch.tensor(np.stack([.5+case_random(c,3,(12,)) for c in self.cases]),device=self.device)*self.default_dof_pos+self.actuator.encoder_bias.to(self.device)
        self.dof_state[:,:,0]=0.;self.dof_state[:,self.gather_indices,0]=q
        self.dof_state[:,:,1]=0.
        self.root_states[:,:3]=self.env_origins
        self.root_states[:,2]+=.6
        if self.terrain_mode!='plane':
            self.root_states[:,:2]+=torch.tensor(np.stack([-1+2*case_random(c,4,(2,)) for c in self.cases]),device=self.device)
        self.root_states[:,7:13]=torch.tensor(np.stack([-.5+case_random(c,5,(6,)) for c in self.cases]),device=self.device)
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_states))
        self.noise=torch.tensor(self.noise_numpy,device=self.device);del self.noise_numpy

    def _observations(self):
        q=self.root_states[:,3:7]
        gravity=torch.zeros((self.num_envs,3),device=self.device);gravity[:,2]=-1
        components=(quat_rotate_inverse(q,self.root_states[:,7:10]),quat_rotate_inverse(q,self.root_states[:,10:13]),quat_rotate_inverse(q,gravity),self.commands[:,:3],self.dof_pos-self.actuator.encoder_bias.to(self.device),self.dof_vel,self.actions)
        raw=torch.cat(components,1)*torch.tensor(self.cfg.observation.scales,device=self.device)
        noise=(2*self.noise[min(self.common_step_counter,2001)]-1)*torch.tensor(self.cfg.observation.noise_half_ranges_after_scaling,device=self.device)
        self.obs_buf=(raw+noise).clamp(-100,100)
        self.privileged_obs_buf=torch.zeros((self.num_envs,353),device=self.device)

    def _push_robots(self):
        velocity=torch.tensor(np.stack([-1+2*case_random(c,6,(2,)) for c in self.cases]),device=self.device)
        old=self.root_states[:,7:9].clone()
        self.root_states[:,7:9]=velocity
        self.base_force_world[:,:2]=self.total_mass*(velocity-old)/self.policy_dt
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_states))

    def _compute_policy_target(self):return self.default_dof_pos+.5*self.actions

    def _step_actuator(self,target):
        q=self.dof_pos;v=self.dof_vel
        safe=pace_v2_hard_limit_safe_target(q,target,self.lower_limits.expand_as(q),self.upper_limits.expand_as(q),self.cfg.action.soft_limit_band_rad)
        result=self.actuator.step(safe,q,v)
        sat=(result.raw_pd_torque-result.saturated_torque).abs()>1e-6
        band=self.cfg.action.soft_limit_band_rad
        active=((q>=self.upper_limits-band)&(target>self.upper_limits))|((q<=self.lower_limits+band)&(target<self.lower_limits))
        exceed=(q<self.lower_limits)|(q>self.upper_limits)
        electrical=.0192*result.applied_torque.square().sum(1)
        signed=(result.applied_torque*v).sum(1)
        self.control+=torch.stack((sat.float().mean(1),active.float().mean(1),exceed.float().mean(1),electrical,signed,signed.clamp_min(0)),1)/4
        self.per_joint+=sat.float()/4
        self.raw_min=torch.minimum(self.raw_min,target);self.raw_max=torch.maximum(self.raw_max,target)
        self.safe_min=torch.minimum(self.safe_min,safe);self.safe_max=torch.maximum(self.safe_max,safe)
        return result

    def step(self,actions):
        n=self.num_envs
        self.control=torch.zeros((n,6),device=self.device);self.per_joint=torch.zeros((n,12),device=self.device)
        self.raw_min=torch.full((n,12),float('inf'),device=self.device);self.raw_max=-self.raw_min
        self.safe_min=self.raw_min.clone();self.safe_max=self.raw_max.clone()
        return super().step(actions)

    def park_inactive(self):
        ids=~self.active
        if ids.any():
            self.root_states[ids]=self.initial_root[ids];self.dof_state[ids]=self.initial_dof[ids]
            indices=ids.nonzero().flatten().to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(indices),len(indices))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(indices),len(indices))
