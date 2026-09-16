import unittest
import torch
from pace_stage1.stage3_cost import EnergyCostSpec,EnergyCostLedger


class Stage3CostChecks(unittest.TestCase):
    def test_power_and_energy_units_and_potential_exclusion(self):
        spec=EnergyCostSpec('diagnostic_nonnegative')
        components=torch.tensor([[100.,50.,-500.],[100.,50.,500.]])
        commands=torch.tensor([[0.,0.,0.],[3.,4.,10.]])
        self.assertTrue(torch.equal(spec.power_w(components,commands),torch.tensor([150.,150.])))
        self.assertTrue(torch.equal(spec.step_cost_j(components,commands),torch.tensor([1.5,1.5])))

    def test_corrected_power_remains_signed_and_uses_planar_command_only(self):
        spec=EnergyCostSpec('pace_corrected')
        components=torch.tensor([[100.,50.,-200.],[100.,50.,50.]])
        commands=torch.tensor([[0.,0.,100.],[1.,0.,100.]])
        self.assertTrue(torch.equal(spec.power_w(components,commands),torch.tensor([-50.,100.])))

    def test_terminal_energy_included_and_only_done_slots_reset(self):
        ledger=EnergyCostLedger(2,'cpu')
        ledger.add(torch.tensor([1.,3.]));ledger.add(torch.tensor([1.,3.]))
        completed=ledger.finish(torch.tensor([0]))
        self.assertEqual(completed['energy_j'].item(),2.)
        self.assertAlmostEqual(completed['duration_s'].item(),.02)
        self.assertAlmostEqual(completed['mean_power_w'].item(),100.)
        self.assertEqual(ledger.energy_j.tolist(),[0.,6.])
        ledger.add(torch.tensor([1.,3.]))
        self.assertAlmostEqual(ledger.mean_power_w().item(),200.)
        self.assertEqual(ledger.steps.tolist(),[1,3])

    def test_time_weighted_mean_is_not_mean_of_episode_means(self):
        ledger=EnergyCostLedger(1,'cpu')
        ledger.add(torch.tensor([1.]));first=ledger.finish(torch.tensor([0]))
        for _ in range(3):ledger.add(torch.tensor([3.]))
        second=ledger.finish(torch.tensor([0]))
        self.assertAlmostEqual(ledger.mean_power_w().item(),250.)
        self.assertAlmostEqual((first['mean_power_w'].item()+second['mean_power_w'].item())/2,200.)


if __name__=='__main__':unittest.main()
