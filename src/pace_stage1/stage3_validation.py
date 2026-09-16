"""有原生日志监督的 Stage3（第三阶段）约束算法短训验证。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT/'artifacts/stage3-cost-calibration/gpt-stage1-pace-corrected-v1/gpt-完整功率标定汇总.json'


def run(args):
    # Simulator import must precede torch.
    from .stage3_env import Stage3CostEnv
    import torch
    from .stage3_lagrangian import ConstraintConfig
    from .stage3_runner import Stage3Runner
    from .pace_v2_runner import validation_train_cfg
    from .pace_v2_formal import save_verified, MILESTONES
    from .pace_v2_validation import write
    from .formal_diagnostics import normalization_summary
    calibration = json.loads(args.calibration.read_text())
    if (calibration['cost_definition'] != 'pace_corrected' or
            calibration['primary']['case_count'] != 28800 or
            calibration['classification'] != 'FIXED_EVALUATION_DISTRIBUTION_COST_CALIBRATION'):
        raise ValueError('complete corrected-power calibration required')
    constraint = ConstraintConfig(reference_power_w=calibration['primary']['time_weighted_mean_w'],
                                  budget_fraction=args.budget_fraction)
    cfg = validation_train_cfg(args.iterations)
    seed = getattr(args, 'seed', 0)
    formal = getattr(args, 'formal_search', False)
    cfg['seed'] = seed
    cfg['runner']['experiment_name'] = 'pace_stage3_fixed_budget_search' if formal else 'pace_stage3_average_power_validation'
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    setup = dict(seed=seed, num_envs=args.num_envs, formal_search=formal)
    manifest = dict(stage='Stage3 单一平均功率约束', classification='固定预算正式搜索，首轮单训练种子' if formal else '算法短训验证，非正式训练',
        state='constructing', iterations=args.iterations, num_envs=args.num_envs,
        seed=seed, train_cfg=cfg, constraint=asdict(constraint), budget_power_w=constraint.budget_power_w,
        training_setup=setup, calibration=str(args.calibration),
        calibration_distribution='固定最终评估分布；不等同于在线训练分布',
        initialization='从头初始化，不继承 Stage1/2 权重',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time())
    if args.resume:
        manifest.update(resume_from=str(args.resume),
            resume_limitation='恢复模型、两套优化器、归一化、乘子和迭代时钟；仿真与全局随机流重新开始，不是逐位轨迹续接')
        previous = output/'gpt-运行记录.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            if old['seed'] != seed or old['num_envs'] != args.num_envs:
                raise ValueError('resume seed/environment count differs from original run')
            shutil.copyfile(previous, output/('gpt-续训前记录-%d.json'%time.time_ns()))
    env = None
    try:
        snapshot = output/('gpt-source-%d'%time.time_ns())
        shutil.copytree(ROOT/'src', snapshot/'src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(args.calibration, snapshot/'gpt-预算标定参照.json')
        write(output/'gpt-运行记录.json', manifest)
        torch.set_num_threads(4)
        env = Stage3CostEnv(cost_definition='pace_corrected', num_envs=args.num_envs, seed=seed)
        runner = Stage3Runner(env, cfg, constraint=constraint, device='cuda:0', log_dir=str(output))
        runner.training_setup = setup
        if args.resume:
            saved = torch.load(args.resume, map_location='cpu').get('infos', {}).get('training_setup')
            if (saved is not None and saved != setup) or (formal and saved is None):
                raise ValueError('resume must match the fixed-budget run; no engineering warm start')
            runner.load(args.resume)
        start = runner.current_learning_iteration
        if start >= args.iterations:
            raise ValueError('target iterations must exceed checkpoint iteration')
        initial_count = runner.normalizers.actor.count.item()
        save_verified(runner, output/('gpt-model-%d.pt'%start))
        manifest.update(state='running', segment_start_iteration=start, gpu_name=torch.cuda.get_device_name(0))
        write(output/'gpt-运行记录.json', manifest)
        with (output/'gpt-逐迭代指标.jsonl').open('a' if args.resume else 'w') as stream:
            for i in range(start, args.iterations):
                env.reset_metrics()
                row = runner.iteration()
                row.update(env.metrics())
                row['normalization'] = normalization_summary(runner.normalizers)
                expected = initial_count + args.num_envs*(1+24*(i-start+1))
                if row['actor_count'] != expected or row['critic_count'] != expected:
                    raise RuntimeError('observation statistics counted incorrectly')
                if env.eq9_calls != (i-start+1)*24*4:
                    raise RuntimeError('actuator substep count mismatch')
                line = json.dumps(row, ensure_ascii=False, allow_nan=False)
                stream.write(line+'\n'); stream.flush(); print(line, flush=True)
                if (i+1)%(500 if formal else 50) == 0 or i+1 == args.iterations or (formal and i+1 in MILESTONES):
                    save_verified(runner, output/('gpt-model-%d.pt'%(i+1)))
                    manifest['completed_iterations'] = i+1
                    write(output/'gpt-运行记录.json', manifest)
        # Check complete constraint state restore, including the next update.
        checkpoint = output/('gpt-model-%d.pt'%args.iterations)
        before = runner.alg.multiplier
        with torch.no_grad():
            observations = env.get_privileged_observations()[:64]
            cost_before = runner.alg.evaluate_cost(observations).clone()
        runner.alg.multiplier = before + 1
        with torch.no_grad():
            next(runner.alg.cost_critic.parameters()).add_(1)
        runner.load(checkpoint)
        with torch.no_grad():
            if runner.alg.multiplier != before or not torch.equal(cost_before, runner.alg.evaluate_cost(observations)):
                raise RuntimeError('cost checkpoint restore mismatch')
        manifest.update(state='completed', completed_iterations=args.iterations,
                        constraint_restore_verified=True, final_constraint=row['constraint'])
    except Exception as error:
        manifest.update(state='failed', error=repr(error))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write(output/'gpt-运行记录.json', manifest)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget-fraction', type=float, default=0.9)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations <= 0 or args.num_envs <= 0:
        parser.error('positive iterations and environment count required')
    if args.worker:
        run(args)
    else:
        from .native_training_supervisor import supervise
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stem = args.output.name + ('-续训-%d'%time.time_ns() if args.resume else '')
        result = supervise(['stdbuf', '-oL', '-eL', sys.executable, '-u', '-m',
            'pace_stage1.stage3_validation', *sys.argv[1:], '--worker'],
            args.output.parent/(stem+'-原生监督.log'),
            args.output.parent/(stem+'-原生监督.json'))
        if result['state'] != 'PROCESS_COMPLETED':
            raise RuntimeError('Stage3 validation failed; inspect native log')


if __name__ == '__main__':
    main()
