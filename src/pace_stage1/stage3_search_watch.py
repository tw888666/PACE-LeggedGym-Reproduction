"""固定预算搜索进度与顺序独立验证；不修改训练中的模型。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .stage3_search_protocol import screen_candidate


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def latest_row(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end-262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def training_status(config):
    statuses = []
    for job in config['training']:
        directory = Path(job['output'])
        manifest = read_json(directory/'gpt-运行记录.json')
        native = read_json(directory.with_name(directory.name+'-原生监督.json'))
        row = latest_row(directory/'gpt-逐迭代指标.jsonl')
        state = manifest.get('state', 'pending')
        if native.get('state') == 'FAILED':
            state = 'failed'
        if state == 'completed' and native.get('state') != 'PROCESS_COMPLETED':
            state = 'finishing'
        statuses.append(dict(name=job['name'], budget_w=job['budget_w'], gpu=job['gpu'],
            state=state, iteration=row.get('next_iteration', 0), constraint=row.get('constraint'),
            task_reward=row.get('reward'), episode=row.get('episode'),
            native_error=native.get('native_error'), error=manifest.get('error')))
    return statuses


def compare(root, config):
    baseline = read_json(root/'validation/gpt-Stage2/gpt-验证汇总.json')
    if not baseline:
        return
    candidates = []
    lines = ['# Stage3（第三阶段）局部线性预算搜索结果', '',
        '首轮各一个训练种子；此表为独立验证上的点估计筛选，不是最终预算或算法优劣结论。', '',
        '| 预算 W（瓦） | 实测修正功率 W | 联合成功率 | 生存率 | 线速度误差 m/s | 偏航误差 rad/s | 初步符合筛选条件 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for job in config['training']:
        summary = read_json(root/'validation'/job['name']/'gpt-验证汇总.json')
        if not summary:
            continue
        primary = summary['primary']
        screened = screen_candidate(primary, baseline['primary'], job['budget_w'])
        strata = []
        for current, reference in zip(summary['strata'], baseline['strata']):
            keys = ('terrain', 'difficulty_row', 'command_bin')
            if any(current[k] != reference[k] for k in keys):
                raise ValueError('candidate and baseline validation strata differ')
            strata.append(dict({k: current[k] for k in keys},
                joint_success_difference=current['joint_task_success']-reference['joint_task_success'],
                survival_difference=current['survival']-reference['survival'],
                corrected_power_difference_w=current['corrected_power_w']-reference['corrected_power_w']))
        candidates.append(dict(name=job['name'], budget_w=job['budget_w'], metrics=primary,
                              screening=screened, stratified_differences_from_stage2=strata))
        lines.append('| %g | %.3f | %.2f%% | %.2f%% | %.4f | %.4f | %s |'%(
            job['budget_w'], primary['corrected_power_w'], 100*primary['joint_task_success'],
            100*primary['survival'], primary['velocity_rmse'], primary['yaw_rmse'],
            '是，待多种子复验' if screened['point_estimate_eligible'] else '否'))
    eligible = [r['budget_w'] for r in candidates if r['screening']['point_estimate_eligible']]
    write(root/'gpt-预算对照结果.json', dict(stage2=baseline['primary'],
        stage1=read_json(root/'validation/gpt-Stage1/gpt-验证汇总.json').get('primary'),
        candidates=candidates, minimum_point_estimate_candidate_w=min(eligible) if eligible else None,
        final_budget_selected=False, training_replication_complete=False))
    lines += ['', '完整地形、难度和指令差异见同目录 JSON（结构化数据）中的分层结果。',
              '下一步复验边界候选的独立训练种子；接近预算或任务容差的点不能仅凭单次均值确定最终 B。']
    (root/'gpt-预算对照结果.md').write_text('\n'.join(lines)+'\n')


def watch(root):
    config = read_json(root/'gpt-搜索配置.json')
    jobs = config['evaluation']
    previous = read_json(root/'gpt-搜索进度.json')
    failed = previous.get('failed_evaluations', {})
    active = None
    while True:
        training = training_status(config)
        by_name = {r['name']: r for r in training}
        if active is not None and active['process'].poll() is not None:
            job = active['job']
            code = active['process'].returncode
            active['stream'].close()
            if code or not (Path(job['output'])/'gpt-验证汇总.json').exists():
                failed[job['name']] = dict(returncode=code, log=active['log'])
            active = None
            compare(root, config)
        completed = [j['name'] for j in jobs if (Path(j['output'])/'gpt-验证汇总.json').exists()]
        if active is None:
            for job in jobs:
                if job['name'] in completed or job['name'] in failed:
                    continue
                if job.get('training_name'):
                    state = by_name[job['training_name']]['state']
                    if state == 'failed':
                        failed[job['name']] = dict(reason='训练失败，未评估该候选')
                        continue
                    if state != 'completed':
                        continue
                log = root/('gpt-%s-独立验证入口.log'%job['name'])
                stream = log.open('a')
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config['evaluation_gpu']))
                process = subprocess.Popen([sys.executable, '-u', '-m', 'pace_stage1.stage3_search_evaluation',
                    '--output', job['output'], '--checkpoint', job['checkpoint'], '--plan', config['validation_plan']],
                    env=environment, stdout=stream, stderr=subprocess.STDOUT)
                active = dict(job=job, process=process, stream=stream, log=str(log))
                break
        finished = all(s['state'] in ('completed', 'failed') for s in training) and len(completed)+len(failed) == len(jobs) and active is None
        progress = dict(state=('completed' if not failed else 'needs_attention') if finished else 'running',
            updated_unix=time.time(), training=training, completed_evaluations=completed,
            active_evaluation=active['job']['name'] if active else None, failed_evaluations=failed)
        if active:
            directory = Path(active['job']['output'])
            progress['active_evaluation_completed_groups'] = len(list(directory.glob('gpt-group-*/gpt-完成记录.json')))
        write(root/'gpt-搜索进度.json', progress)
        if finished:
            compare(root, config)
            return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    watch(args.root.resolve())
