"""Independent deterministic milestone evaluation; never updates the training policy."""
from .pace_v2_formal import FormalEnv, FormalRunner, MILESTONES

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from .config import STAGE1_CONFIG
from .pace_v2_runner import validation_train_cfg
from .pace_v2_validation import write
from .formal_diagnostics import normalization_summary


def evaluate(checkpoint, output):
    torch.set_num_threads(4)
    cfg = replace(STAGE1_CONFIG, terrain=replace(STAGE1_CONFIG.terrain, curriculum=False))
    env = FormalEnv(num_envs=64, seed=100000, config=cfg)
    try:
        runner = FormalRunner(env, validation_train_cfg(30000), device='cuda:0')
        runner.load(checkpoint, load_optimizer=False)
        model = runner.alg.actor_critic
        model.eval()
        env.set_training_iteration(max(0,runner.current_learning_iteration-1))
        initial_norm = normalization_summary(runner.normalizers)
        env.reset_metrics()
        episodes=[]
        with torch.inference_mode():
            for _ in range(2000):
                obs = env.get_observations()
                actions = model.act_inference(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError('non-finite deterministic action')
                _,_,reward,_,info = env.step(actions)
                if not torch.isfinite(reward).all():
                    raise RuntimeError('non-finite evaluation reward')
                if info.get('episode'): episodes.append(info['episode'])
        if normalization_summary(runner.normalizers) != initial_norm:
            raise RuntimeError('evaluation changed normalizer statistics')
        result = dict(checkpoint=str(checkpoint), next_iteration=runner.current_learning_iteration,
                      classification='INDEPENDENT DETERMINISTIC CHECKPOINT EVALUATION',
                      num_envs=64, seed=100000, policy_steps=2000, terrain='trimesh',
                      terrain_curriculum=False, policy='actor mean; normalizer eval mode',
                      observation_noise='retained preregistered environment noise',
                      state='completed', **env.metrics())
        if episodes:
            result['completed_episode_statistics']={k:sum(float(e[k]) for e in episodes if k in e)/sum(k in e for e in episodes) for k in set().union(*(e.keys() for e in episodes))}
        result['normalization_frozen']=True
        write(output,result)
    finally:
        env.close()


def watch(run_dir):
    output = run_dir / 'gpt-独立评估'
    output.mkdir(parents=True, exist_ok=True)
    for milestone in MILESTONES:
        checkpoint = run_dir / ('gpt-model-%d.pt' % milestone)
        target = output / ('gpt-evaluation-%d.json' % milestone)
        if target.exists():
            continue
        while not checkpoint.exists():
            manifest = run_dir / 'gpt-运行记录.json'
            if manifest.exists() and json.loads(manifest.read_text())['state']=='failed':
                raise RuntimeError('training failed before next evaluation milestone')
            time.sleep(15)
        evaluate(checkpoint,target)
        print('completed deterministic evaluation',milestone,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    watch(parser.parse_args().run_dir)
