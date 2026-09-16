"""GPU reward wiring test with fixed actions only: never constructs PPO."""
import os
import unittest


@unittest.skipUnless(os.environ.get('PACE_STAGE2_WIRING_TEST')=='1','explicit GPU wiring test')
class Stage2WiringTest(unittest.TestCase):
    def test_reward_clock_and_terminal_accounting(self):
        from pace_stage1.stage2_formal import Stage2Env
        from pace_stage1.semantics import foot_touchdown_schedule
        import torch
        env_class=Stage2Env
        if os.environ.get('PACE_TEST_CONSTRUCTION_REPAIR')=='1':
            from pace_stage1.construction_staging import CapacitySafeConstructionMixin
            class RepairedEnv(CapacitySafeConstructionMixin,Stage2Env):pass
            env_class=RepairedEnv
        env=env_class(num_envs=8,potential_sign=-1)
        try:
            action=torch.zeros((8,12),device=env.device)
            env.set_training_iteration(0)
            *_,info=env.step(action)
            self.assertTrue(torch.equal(info['reward_terms']['scaled_energy'],torch.zeros(8,device=env.device)))
            env.set_training_iteration(500)
            # Force timeout to exercise accounting before command/reset mutation.
            env.episode_length_buf[:]=env.max_episode_length
            _,_,reward,done,info=env.step(action)
            terms=info['reward_terms']
            task=.01*(.2*terms['velocity_tracking']-terms['collision']-.1*foot_touchdown_schedule(500)*terms['foot_touchdown'])
            expected_delta=-.01*16e-5*.5*env._energy_power.sum(1)*terms['energy_gamma_v']
            self.assertTrue(torch.allclose(reward-task,expected_delta,atol=1e-8))
            self.assertTrue(done.all())
            self.assertIn('rew_energy',info['episode'])
            components=sum(info['episode']['rew_'+k] for k in ('velocity_tracking','collision','foot_touchdown','termination','energy'))
            self.assertTrue(torch.allclose(info['episode']['rew_total'],components,atol=1e-7))
            self.assertTrue(torch.equal(env.episode_sums['energy'],torch.zeros(8,device=env.device)))
            self.assertEqual(env.eq9_calls,8)
        finally:
            env.close()


if __name__=='__main__':unittest.main()
