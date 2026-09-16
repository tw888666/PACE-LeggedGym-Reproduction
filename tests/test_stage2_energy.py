import unittest
import torch
from pace_stage1.stage2_energy import energy_components, energy_reward


class EnergyTest(unittest.TestCase):
    def test_mechanical_rectification_is_after_joint_sum(self):
        torque=torch.tensor([[10.,10.]])
        power=energy_components(torque,torch.tensor([[1.,-2.]]),torch.tensor([2.]),torch.zeros(1,1),potential_sign=-1)
        self.assertAlmostEqual(power[0,0].item(),3.84,places=5)
        self.assertEqual(power[0,1].item(),0.)

    def test_uphill_interpretation_rewards_potential_gain(self):
        power=energy_components(torch.zeros(1,2),torch.zeros(1,2),torch.tensor([2.]),torch.ones(1,1),potential_sign=-1)
        delta,_,_=energy_reward(power,torch.zeros(1,3),500)
        self.assertGreater(delta.item(),0.)
        reverse=energy_components(torch.zeros(1,2),torch.zeros(1,2),torch.tensor([2.]),-torch.ones(1,1),potential_sign=-1)
        self.assertTrue(torch.equal(reverse,-power))

    def test_half_life_units_and_linear_command_normalization(self):
        power=torch.tensor([[1000.,0.,0.],[1000.,0.,0.]])
        commands=torch.tensor([[0.,0.,10.],[1.,0.,0.]])
        zero,_,_=energy_reward(power,commands,0)
        self.assertTrue(torch.equal(zero,torch.zeros(2)))
        half,_,gamma=energy_reward(power,commands,500)
        self.assertTrue(torch.allclose(half,torch.tensor([-.0008,-.0004])))
        self.assertTrue(torch.equal(gamma,torch.tensor([1.,.5])))

    def test_sign_is_not_implicit(self):
        with self.assertRaises(ValueError):
            energy_components(torch.zeros(1,2),torch.zeros(1,2),torch.ones(1),torch.zeros(1,1),potential_sign=None)


if __name__=='__main__':unittest.main()
