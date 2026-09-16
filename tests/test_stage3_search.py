import csv
import unittest
import json
import tempfile
from pathlib import Path

from pace_stage1.stage3_formal import budget_fraction
from pace_stage1.stage3_search_protocol import ROOT, BUDGETS_W, validation_cases, aggregate, screen_candidate


class Stage3SearchChecks(unittest.TestCase):
    def test_watt_budgets_keep_the_same_normalization_reference(self):
        calibration = dict(cost_definition='pace_corrected', primary=dict(time_weighted_mean_w=1958.0968551728386))
        for budget in BUDGETS_W:
            ratio = budget_fraction(budget, calibration)
            self.assertAlmostEqual(ratio*calibration['primary']['time_weighted_mean_w'], budget)
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                budget_fraction(invalid, calibration)

    def test_independent_cases_preserve_balanced_coverage_and_commands(self):
        with (ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv').open() as stream:
            original = list(csv.DictReader(stream))
        cases = validation_cases(original)
        self.assertEqual(len(cases), 3672)
        self.assertEqual(len({c['case_id'] for c in cases}), 3672)
        self.assertFalse({c['case_id'] for c in cases} & {c['case_id'] for c in original})
        self.assertEqual({c['seed'] for c in cases}, {'210000', '210001', '210002'})
        primary = [c for c in cases if c['terrain'] != 'flat_reference']
        self.assertEqual(len(primary), 3600)
        from collections import Counter
        strata = Counter((c['terrain'], c['difficulty_row'], c['command_bin'], c['seed']) for c in primary)
        self.assertEqual(len(strata), 450)
        self.assertEqual(set(strata.values()), {8})
        old = {int(c['case_id']): c for c in original}
        for c in cases:
            for key in ('terrain', 'difficulty_row', 'source_column', 'command_bin', 'vx_m_s', 'vy_m_s', 'yaw_rad_s'):
                self.assertEqual(c[key], old[int(c['case_id'])-1000000][key])

    def test_power_uses_elapsed_time_and_failure_is_retained(self):
        first = dict(duration_s=1., corrected_cost_j=-10., survival=False,
                     joint_task_success=False, velocity_rmse=.4, yaw_rmse=.2)
        second = dict(duration_s=3., corrected_cost_j=210., survival=True,
                      joint_task_success=True, velocity_rmse=.1, yaw_rmse=.1)
        result = aggregate([first, second])
        self.assertEqual(result['corrected_power_w'], 50.)
        self.assertEqual(result['survival'], .5)
        self.assertAlmostEqual(result['velocity_rmse'], .25)

    def test_low_energy_does_not_override_task_or_tracking_failure(self):
        base = dict(corrected_power_w=138., joint_task_success=.96, survival=.98,
                    velocity_rmse=.1, yaw_rmse=.06)
        candidate = dict(base, corrected_power_w=120.)
        self.assertTrue(screen_candidate(candidate, base, 125)['point_estimate_eligible'])
        self.assertFalse(screen_candidate(candidate, base, 100)['point_estimate_eligible'])
        for changed in (dict(joint_task_success=.90), dict(survival=.95),
                        dict(velocity_rmse=.2), dict(yaw_rmse=.1)):
            result = screen_candidate(dict(candidate, **changed), base, 125)
            self.assertFalse(result['point_estimate_eligible'])
            self.assertFalse(result['final_budget_selected'])

    def test_progress_ignores_partial_lines_and_requires_native_completion(self):
        from pace_stage1.stage3_search_watch import latest_row, training_status
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'gpt-B125'
            directory.mkdir()
            stream = directory/'gpt-逐迭代指标.jsonl'
            stream.write_text('{"next_iteration": 29999}\n{"next_iteration":')
            self.assertEqual(latest_row(stream)['next_iteration'], 29999)
            (directory/'gpt-运行记录.json').write_text('{"state":"completed"}')
            config = dict(training=[dict(output=str(directory), name='gpt-B125', budget_w=125, gpu=0)])
            self.assertEqual(training_status(config)[0]['state'], 'finishing')
            native = directory.with_name(directory.name+'-原生监督.json')
            native.write_text('{"state":"FAILED","native_error":"physics error"}')
            self.assertEqual(training_status(config)[0]['state'], 'failed')

    def test_report_selects_only_a_provisional_eligible_budget(self):
        from pace_stage1.stage3_search_watch import compare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = dict(corrected_power_w=120., joint_task_success=.97, survival=.99,
                           velocity_rmse=.1, yaw_rmse=.05)
            jobs = []
            for name, budget in (('gpt-Stage2', None), ('gpt-B100', 100), ('gpt-B125', 125)):
                directory = root/'validation'/name
                directory.mkdir(parents=True)
                (directory/'gpt-验证汇总.json').write_text(json.dumps(dict(primary=metrics, strata=[])))
                if budget:
                    jobs.append(dict(name=name, budget_w=budget))
            compare(root, dict(training=jobs))
            result = json.loads((root/'gpt-预算对照结果.json').read_text())
            self.assertEqual(result['minimum_point_estimate_candidate_w'], 125)
            self.assertFalse(result['final_budget_selected'])


if __name__ == '__main__':
    unittest.main()
