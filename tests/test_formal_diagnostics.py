import unittest
import torch
from pace_stage1.formal_diagnostics import action_tail_summary, normalization_summary
from pace_stage1.empirical_normalization import ActorCriticEmpiricalNormalizers

class FormalDiagnosticsTest(unittest.TestCase):
    def test_exact_tails_without_input_or_rng_mutation(self):
        x=torch.tensor([[0.,2.,5.,10.],[3.,6.,11.,20.]])
        original=x.clone();rng=torch.random.get_rng_state().clone()
        result=action_tail_summary([x[:1],x[1:]])
        self.assertEqual(result['prob_abs_gt_2'],6/8)
        self.assertEqual(result['prob_abs_gt_5'],4/8)
        self.assertEqual(result['prob_abs_gt_10'],2/8)
        self.assertEqual(result['per_joint_prob_abs_gt_10'],[0.,0.,.5,.5])
        self.assertAlmostEqual(result['p99'],torch.quantile(x.flatten(),.99).item())
        self.assertTrue(torch.equal(x,original))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
    def test_corrupt_normalizer_is_rejected(self):
        norms=ActorCriticEmpiricalNormalizers(48,353)
        before=normalization_summary(norms)
        self.assertEqual(before['actor']['count'],0)
        norms.actor._var.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError,'non-finite'):normalization_summary(norms)
