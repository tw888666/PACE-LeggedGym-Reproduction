"""Stage3（第三阶段）固定瓦数预算正式训练，复用已验证采样与算法。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .stage3_validation import DEFAULT_CALIBRATION, run
from .native_training_supervisor import supervise


def budget_fraction(budget_w, calibration):
    if not math.isfinite(budget_w) or budget_w <= 0:
        raise ValueError('positive finite power budget required')
    if calibration['cost_definition'] != 'pace_corrected':
        raise ValueError('corrected-power calibration required')
    reference = calibration['primary']['time_weighted_mean_w']
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('positive finite reference required')
    return budget_w/reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-w', type=float, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-iterations', type=int, help='共享入口冒烟测试，64 环境，标为非正式')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.budget_fraction = budget_fraction(args.budget_w, json.loads(args.calibration.read_text()))
    args.formal_search = args.smoke_iterations is None
    args.iterations = 30000 if args.formal_search else args.smoke_iterations
    args.num_envs = 4096 if args.formal_search else 64
    if args.iterations <= 0 or args.seed < 0:
        parser.error('positive iterations and nonnegative seed required')
    if args.worker:
        run(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.name + ('-续训-%d'%time.time_ns() if args.resume else '')
    result = supervise(['stdbuf', '-oL', '-eL', sys.executable, '-u', '-m',
        'pace_stage1.stage3_formal', *sys.argv[1:], '--worker'],
        args.output.parent/(stem+'-原生监督.log'), args.output.parent/(stem+'-原生监督.json'))
    if result['state'] != 'PROCESS_COMPLETED':
        raise RuntimeError('fixed-budget training failed; inspect native log')
    manifest = json.loads((args.output/'gpt-运行记录.json').read_text())
    if manifest['state'] != 'completed' or manifest['completed_iterations'] != args.iterations:
        raise RuntimeError('training did not complete its declared iteration budget')


if __name__ == '__main__':
    main()
