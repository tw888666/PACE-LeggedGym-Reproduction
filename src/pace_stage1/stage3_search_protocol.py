"""固定预算搜索的独立验证样本、聚合和初步候选判定。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUDGETS_W = (100, 125, 150, 175, 200)
VALIDATION_PLAN = ROOT/'provenance/gpt-Stage3-预算搜索独立验证样本-v1.csv'


def validation_cases(final_cases):
    """Same coverage, eight repeats, independent seeds and random-stream IDs."""
    result = []
    for original in final_cases:
        if int(original['replicate']) >= 8:
            continue
        case = dict(original)
        case['case_id'] = str(1000000+int(original['case_id']))
        case['seed'] = str(210000+int(original['seed'])-200000)
        result.append(case)
    return result


def read_groups(plan=VALIDATION_PLAN):
    with Path(plan).open() as stream:
        cases = list(csv.DictReader(stream))
    keys = list(dict.fromkeys((c['terrain'], c['difficulty_row']) for c in cases))
    return [[c for c in cases if (c['terrain'], c['difficulty_row']) == key] for key in keys]


def aggregate(rows):
    duration = sum(r['duration_s'] for r in rows)
    result = dict(case_count=len(rows), duration_s=duration,
                  corrected_power_w=sum(r['corrected_cost_j'] for r in rows)/duration)
    for name in ('survival', 'joint_task_success', 'velocity_rmse', 'yaw_rmse'):
        result[name] = sum(r[name] for r in rows)/len(rows)
    return result


def summarize(rows):
    primary = [r for r in rows if r['terrain'] != 'flat_reference']
    dimensions = ('terrain', 'difficulty_row', 'command_bin')
    strata = []
    keys = list(dict.fromkeys(tuple(r[k] for k in dimensions) for r in primary))
    for key in keys:
        subset = [r for r in primary if tuple(r[k] for k in dimensions) == key]
        strata.append(dict(zip(dimensions, key), **aggregate(subset)))
    return dict(primary=aggregate(primary), strata=strata,
                flat_reference=aggregate([r for r in rows if r['terrain'] == 'flat_reference']))


def screen_candidate(candidate, baseline, budget_w):
    """Point-estimate screen only; finalists require training-seed replication."""
    checks = dict(
        power=candidate['corrected_power_w'] <= budget_w,
        absolute_task=candidate['joint_task_success'] >= .8 and candidate['survival'] >= .9,
        task_retention=candidate['joint_task_success'] >= baseline['joint_task_success']-.02,
        survival_retention=candidate['survival'] >= baseline['survival']-.01,
        linear_tracking=candidate['velocity_rmse'] <= 1.1*baseline['velocity_rmse'],
        yaw_tracking=candidate['yaw_rmse'] <= 1.1*baseline['yaw_rmse'])
    return dict(checks=checks, point_estimate_eligible=all(checks.values()),
                final_budget_selected=False)
