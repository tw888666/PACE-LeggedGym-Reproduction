"""Fixed-action GPU wiring check, with no PPO or policy update."""
import os
import unittest


@unittest.skipUnless(os.environ.get('PACE_STAGE3_WIRING_TEST')=='1','explicit GPU wiring test')
class Stage3EnvironmentChecks(unittest.TestCase):
    def test_task_reward_and_terminal_cost(self):
        self.check_cost('diagnostic_nonnegative')

    def test_primary_corrected_cost_matches_stage2_energy_before_reset(self):
        self.check_cost('pace_corrected')

    def check_cost(self,definition):
        from pace_stage1.stage3_env import Stage3CostEnv
        from pace_stage1.semantics import foot_touchdown_schedule
        import torch
        from pace_stage1.stage2_energy import energy_reward
        env=Stage3CostEnv(cost_definition=definition,num_envs=8)
        try:
            env.set_training_iteration(500)
            env.reset_metrics()
            actions=torch.zeros((8,12),device=env.device)
            env.step(actions)
            env.episode_length_buf[:]=env.max_episode_length
            _,_,reward,done,info=env.step(actions)
            terms=info['reward_terms']
            task=.01*(.2*terms['velocity_tracking']-terms['collision']-.1*foot_touchdown_schedule(500)*terms['foot_touchdown'])
            torch.testing.assert_close(reward,task,atol=1e-8,rtol=1e-5)
            self.assertNotIn('scaled_energy',terms)
            if definition=='diagnostic_nonnegative':
                self.assertTrue(torch.allclose(info['cost_power_w'],info['cost_components_w'][:,:2].sum(1)))
            else:
                original_energy,_,_=energy_reward(info['cost_components_w'],info['cost_commands'],500,env.policy_dt)
                self.assertTrue(torch.allclose(-16e-5*.5*info['costs'],original_energy,atol=1e-9))
            self.assertTrue(torch.equal(info['costs'],.01*info['cost_power_w']))
            self.assertTrue(torch.isfinite(info['costs']).all())
            if definition=='diagnostic_nonnegative':self.assertTrue((info['costs']>=0).all())
            self.assertTrue(done.all())
            self.assertEqual(len(info['cost_episodes']['env_ids']),8)
            self.assertTrue(torch.equal(env.cost_ledger.steps,torch.zeros_like(env.cost_ledger.steps)))
            if definition=='diagnostic_nonnegative':self.assertTrue((info['cost_episodes']['energy_j']>=info['costs']).all())
            self.assertNotIn('rew_energy',info['episode'])
            self.assertEqual(env.eq9_calls,8)
            diagnostic=env.metrics()['energy_logging_only']['electrical_plus_positive_net_w']
            if definition=='diagnostic_nonnegative':self.assertAlmostEqual(env.cost_ledger.mean_power_w().item(),diagnostic,places=3)
        finally:env.close()


if __name__=='__main__':unittest.main()
