"""Gated Stage1 PACE-v2 reconstruction: 4096 mixed-terrain environments, seed 0."""
# Isaac Gym must be imported before torch.
from .env import Stage1LocomotionEnv

import copy
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import torch
from rsl_rl.runners import OnPolicyRunner

from .formal_diagnostics import FormalDiagnosticsMixin, normalization_summary
from .config import STAGE1_CONFIG
from .normalization_bridge import NormalizationRunnerMixin
from .pace_v2_control import PaceV2ControlMixin
from .pace_v2_profiles import pace_v2_formal_profile
from .pace_v2_rewards import pace_v2_environment_config
from .pace_v2_runner import ValidationLoopMixin, validation_train_cfg
from .pace_v2_validation import InstrumentationMixin, write
from .protocol_gate import require_formal_training_allowed, PROJECT_ROOT


class FormalEnv(FormalDiagnosticsMixin, InstrumentationMixin, PaceV2ControlMixin, Stage1LocomotionEnv):
    def __init__(self, num_envs=4096, seed=0, config=STAGE1_CONFIG):
        self.protocol_profile = pace_v2_formal_profile()
        super().__init__(config=pace_v2_environment_config(config),
                         num_envs=num_envs, seed=seed, terrain_mode='trimesh',
                         sim_device='cuda:0', headless=True)


class FormalRunner(ValidationLoopMixin, NormalizationRunnerMixin, OnPolicyRunner):
    def __init__(self, env, *args, **kwargs):
        require_formal_training_allowed()
        if env.protocol_profile.identity != 'PACE_V2_FORMAL':
            raise ValueError('formal runner requires formal environment')
        super().__init__(env, *args, **kwargs)


MILESTONES = (0,300,1000,3000,5000,10000,15000,20000,25000,30000)


def save_verified(runner, path):
    temporary = path.with_suffix('.tmp')
    runner.save(temporary, infos={'profile':'PACE_V2_FORMAL'})
    state = torch.load(temporary, map_location=runner.device)
    if state['iter'] != runner.current_learning_iteration or state['entropy_schedule']['last_iteration'] != state['iter']-1:
        raise RuntimeError('checkpoint iteration corruption')
    model = runner.alg.actor_critic
    # Copy only inference modules: PPO's cached distribution can contain non-leaf tensors.
    actor = copy.deepcopy(model.actor).eval()
    critic_model = copy.deepcopy(model.critic).eval()
    actor.load_state_dict({k[6:]:v for k,v in state['model_state_dict'].items() if k.startswith('actor.')}, strict=True)
    critic_model.load_state_dict({k[7:]:v for k,v in state['model_state_dict'].items() if k.startswith('critic.')}, strict=True)
    obs = runner.env.env.get_observations()[:64]
    critic = runner.env.env.get_privileged_observations()[:64]
    with torch.no_grad():
        for a,b in ((model.actor[0](obs),actor[0](obs)),
                    (model.critic[0](critic),critic_model[0](critic)),
                    (model.act_inference(obs),actor(obs)),
                    (model.evaluate(critic),critic_model(critic))):
            if not torch.equal(a,b):
                raise RuntimeError('checkpoint output restore mismatch')
    os.replace(temporary,path)


def run(output, resume=None, construction_capacity_repair=False):
    require_formal_training_allowed()  # Before directory creation or simulator construction.
    profile = pace_v2_formal_profile()
    output.mkdir(parents=True, exist_ok=resume is not None)
    cfg = validation_train_cfg(profile.requested_iterations)
    cfg['seed'] = 0
    cfg['runner']['experiment_name'] = 'pace_v2_formal_stage1_task_only'
    manifest = dict(classification='PACE_V2_FORMAL / PREREGISTERED RECONSTRUCTION',
                    stage='Stage1 Task-only', formal=True, num_envs=4096, seed=0,
                    iterations=30000, terrain='trimesh', train_cfg=cfg,
                    environment_config=asdict(pace_v2_environment_config(STAGE1_CONFIG)),
                    rsl_rl_version=importlib.metadata.version('rsl-rl'), torch_version=torch.__version__,
                    initialization='fresh; no validation checkpoint warm start',
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    started_unix=time.time(), state='constructing')
    if construction_capacity_repair:
        manifest['physics_construction_repair']=dict(separate_construction_only=True,max_gpu_contact_pairs=2**25,
                                                   default_buffer_size_multiplier=5.,changes_learning_protocol=False)
        if resume is not None and json.loads((output/'gpt-运行记录.json').read_text()).get('physics_construction_repair')!=manifest['physics_construction_repair']:
            raise ValueError('cannot resume an unrepaired run into the repaired protocol')
    if resume is not None:
        previous = json.loads((output / 'gpt-运行记录.json').read_text())
        write(output / 'gpt-中断前运行记录.json', previous)
        manifest['resume_from'] = str(resume)
        manifest['resume_limitation'] = 'weights/optimizer/normalizers/clock restored; simulator and RNG restarted; not bitwise trajectory continuation'
    env = None
    try:
        # A readable source copy captures uncommitted implementation too, without requiring a commit.
        snapshot = output / ('gpt-source' if resume is None else 'gpt-source-resume-300')
        shutil.copytree(PROJECT_ROOT / 'src', snapshot / 'src',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(PROJECT_ROOT / 'provenance', snapshot / 'provenance')
        write(output / 'gpt-运行记录.json', manifest)
        torch.set_num_threads(4)
        env_class=FormalEnv
        if construction_capacity_repair:
            from .construction_staging import CapacitySafeConstructionMixin
            class RepairedEnv(CapacitySafeConstructionMixin,FormalEnv):pass
            env_class=RepairedEnv
        env = env_class()
        runner = FormalRunner(env, cfg, log_dir=str(output), device='cuda:0')
        if resume is None:
            save_verified(runner, output / 'gpt-model-0.pt')
        else:
            runner.load(resume)
            save_verified(runner, output / ('gpt-model-%d.pt' % runner.current_learning_iteration))
        start_iteration = runner.current_learning_iteration
        initial_count = runner.normalizers.actor.count.item()
        manifest['segment_start_iteration'] = start_iteration
        manifest.update(state='running', gpu_name=torch.cuda.get_device_name(0))
        write(output / 'gpt-运行记录.json', manifest)
        with (output / 'gpt-逐迭代指标.jsonl').open('w' if resume is None else 'a') as stream:
            for i in range(start_iteration, profile.requested_iterations):
                env.reset_metrics()
                row = runner.iteration()
                row.update(env.metrics())
                row['normalization'] = normalization_summary(runner.normalizers)
                row['segment_start_iteration'] = start_iteration
                row['warnings'] = []
                if row['torque_saturation_ratio'] > .5:
                    row['warnings'].append('torque saturation > 50%; monitor only')
                if row['entropy_coef'] != runner.entropy_scheduler.coefficient(i):
                    raise RuntimeError('entropy injection mismatch')
                if row['actor_count'] != initial_count+4096*(1+24*(i-start_iteration+1)) or row['critic_count'] != row['actor_count']:
                    raise RuntimeError('normalizer sample count mismatch')
                if env.eq9_calls != (i-start_iteration+1)*24*4:
                    raise RuntimeError('Eq9 substep count mismatch')
                line = json.dumps(row, allow_nan=False)
                stream.write(line+'\n')
                stream.flush()
                print(line, flush=True)
                if (i+1) % 500 == 0 or (i+1) in MILESTONES:
                    save_verified(runner, output / ('gpt-model-%d.pt' % (i+1)))
                    manifest['completed_iterations'] = i+1
                    write(output / 'gpt-运行记录.json', manifest)
        shutil.copyfile(output / 'gpt-model-30000.pt', output / 'gpt-model-final.pt')
        manifest.update(state='completed', completed_iterations=30000)
    except Exception as error:
        manifest.update(state='failed', error=repr(error))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write(output / 'gpt-运行记录.json', manifest)
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--construction-capacity-repair',action='store_true')
    args = parser.parse_args()
    run(args.output, args.resume,args.construction_capacity_repair)
