"""独立验证：同一首次回合同时采集任务结果和有符号完整代价。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .stage3_search_protocol import VALIDATION_PLAN, read_groups, summarize
from .native_training_supervisor import supervise


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def replay(cases, directory, checkpoint):
    from .final_evaluation_env import FinalEvaluationEnv
    from .final_evaluation import load_policy
    from .stage2_energy import energy_components
    from .stage3_cost import EnergyCostSpec
    from .semantics import quat_rotate_inverse
    import torch

    class CostEvaluationEnv(FinalEvaluationEnv):
        def _step_actuator(self, target):
            result = super()._step_actuator(target)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.components += energy_components(result.applied_torque, self.dof_vel,
                self.body_mass, self.rigid_body_state[:, :, 9], potential_sign=-1)/4
            return result

    torch.set_num_threads(4)
    started = time.time()
    env = CostEvaluationEnv(cases, directory.parent/'gpt-地形缓存')
    try:
        model, norms = load_policy(checkpoint, env.device)
        norm_before = {k: v.clone() for k, v in norms.state_dict().items()}
        spec = EnergyCostSpec('pace_corrected', env.policy_dt)
        n = env.num_envs
        count = torch.zeros(n, dtype=torch.long, device=env.device)
        totals = torch.zeros(n, 6, dtype=torch.float64, device=env.device)
        error = torch.zeros(n, 2, dtype=torch.float64, device=env.device)
        negative = torch.zeros_like(count)
        survival = torch.zeros(n, dtype=torch.bool, device=env.device)
        with torch.no_grad():
            for step in range(2001):
                active = env.active.clone()
                action = model.act_inference(env.get_observations())
                if not torch.isfinite(action[active]).all():
                    raise RuntimeError('non-finite validation action')
                action[~active] = 0
                env.components = torch.zeros(n, 3, device=env.device)
                _, _, _, done, _ = env.step(action)
                power = spec.power_w(env.components, env.commands)
                values = torch.cat((env.components, power[:, None],
                    power.clamp_min(0)[:, None], power.clamp_max(0)[:, None]), 1)
                quaternion = env.root_states[:, 3:7]
                linear = quat_rotate_inverse(quaternion, env.root_states[:, 7:10])
                angular = quat_rotate_inverse(quaternion, env.root_states[:, 10:13])
                squared = torch.stack(((linear[:, :2]-env.commands[:, :2]).square().sum(1),
                    (angular[:, 2]-env.commands[:, 2]).square()), 1)
                if not torch.isfinite(values[active]).all() or not torch.isfinite(squared[active]).all():
                    raise RuntimeError('non-finite validation cost/state')
                totals += torch.where(active[:, None], values.double(), 0)*env.policy_dt
                error += torch.where(active[:, None], squared.double(), 0)
                count += active
                negative += active & (power < 0)
                contact = (env.contact_forces[:, env.base_indices, :].norm(dim=2) >
                           env.cfg.rewards.termination_contact_threshold_n).any(1)
                survival |= active & done.bool() & env.time_out_buf & ~contact
                displacement = (env.root_states[:, :2]-env.env_origins[:, :2]).abs().amax(1)
                if env.terrain_mode != 'plane' and (active & (displacement > 34)).any():
                    raise RuntimeError('active validation case left the homogeneous terrain patch')
                env.active &= ~done.bool()
                env.park_inactive()
                if (step+1)%200 == 0:
                    write(directory/'gpt-执行进度.json', dict(policy_steps=step+1,
                        active_cases=int(env.active.sum()), seconds=time.time()-started))
                if not env.active.any():
                    break
        if env.active.any():
            raise RuntimeError('validation horizon ended with unfinished cases')
        if not all(torch.equal(norm_before[k], v) for k, v in norms.state_dict().items()):
            raise RuntimeError('validation updated normalization')
        errors = (error/count[:, None]).sqrt().cpu().tolist()
        records = []
        for i, case in enumerate(cases):
            row = dict(case, steps=int(count[i]), duration_s=int(count[i])*env.policy_dt,
                velocity_rmse=errors[i][0], yaw_rmse=errors[i][1], survival=bool(survival[i]),
                joint_task_success=bool(survival[i]) and max(errors[i]) <= .3,
                negative_step_count=int(negative[i]))
            row.update(zip(('electrical_j', 'mechanical_positive_net_j', 'potential_j',
                'corrected_cost_j', 'positive_corrected_j', 'negative_corrected_j'), totals[i].cpu().tolist()))
            row['corrected_mean_w'] = row['corrected_cost_j']/row['duration_s']
            records.append(row)
        write(directory/'gpt-逐回合验证.json', records)
        write(directory/'gpt-完成记录.json', dict(state='completed', case_count=len(records),
            checkpoint=str(checkpoint.resolve()), normalization_frozen=True, seconds=time.time()-started,
            cost_definition='pace_corrected', task_and_cost_same_trajectory=True))
    finally:
        env.close()


def finish(output, checkpoint, plan):
    groups = read_groups(plan)
    rows = []
    for i, cases in enumerate(groups):
        directory = output/('gpt-group-%02d'%i)
        marker = json.loads((directory/'gpt-完成记录.json').read_text())
        if not marker.get('runtime_log_checked') or marker['checkpoint'] != str(checkpoint.resolve()):
            raise ValueError('unaudited or mismatched validation checkpoint')
        group_rows = json.loads((directory/'gpt-逐回合验证.json').read_text())
        if len(group_rows) != len(cases):
            raise ValueError('validation case count mismatch')
        for row, case in zip(group_rows, cases):
            if any(row[k] != v for k, v in case.items()):
                raise ValueError('validation case plan mismatch')
        rows.extend(group_rows)
    summary = summarize(rows)
    summary.update(state='completed', checkpoint=str(checkpoint.resolve()), plan=str(plan.resolve()),
        classification='固定预算选择验证，非最终测试', training_seeds=1,
        cost_definition='pace_corrected', final_budget_selected=False)
    write(output/'gpt-验证汇总.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=VALIDATION_PLAN)
    parser.add_argument('--group', type=int)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    groups = read_groups(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        replay(groups[args.group], args.output/('gpt-group-%02d'%args.group), args.checkpoint)
        return
    selected = range(len(groups)) if args.group is None else [args.group]
    for i in selected:
        directory = args.output/('gpt-group-%02d'%i)
        directory.mkdir(exist_ok=True)
        marker_path = directory/'gpt-完成记录.json'
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker['checkpoint'] != str(args.checkpoint.resolve()):
                raise ValueError('validation output already belongs to another checkpoint')
            if marker.get('runtime_log_checked'):
                continue
        token = str(time.time_ns())
        result = supervise(['stdbuf', '-oL', '-eL', sys.executable, '-u', '-m',
            'pace_stage1.stage3_search_evaluation', '--output', str(args.output),
            '--checkpoint', str(args.checkpoint), '--plan', str(args.plan), '--group', str(i), '--worker'],
            directory/('gpt-原生验证-%s.log'%token), directory/('gpt-原生验证-%s.json'%token))
        if result['state'] != 'PROCESS_COMPLETED':
            raise RuntimeError('validation failed; inspect '+str(directory))
        marker = json.loads(marker_path.read_text())
        marker['runtime_log_checked'] = True
        write(marker_path, marker)
        print('已完成独立验证分组', i, flush=True)
    if args.group is None:
        finish(args.output, args.checkpoint, args.plan)


if __name__ == '__main__':
    main()
