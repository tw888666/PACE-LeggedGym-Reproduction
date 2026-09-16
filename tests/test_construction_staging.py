import unittest
import numpy as np
from pace_stage1.construction_staging import ConstructionStagingMixin,CapacitySafeConstructionMixin,separated_construction_origins


class ConstructionStagingTest(unittest.TestCase):
    def test_capacity_changes_only_contact_limit_and_restores_handle(self):
        from types import SimpleNamespace
        params=SimpleNamespace(physx=SimpleNamespace(max_gpu_contact_pairs=2**23,default_buffer_size_multiplier=5.,contact_offset=.01))
        class Gym:
            def create_sim(self,device,graphics,engine,p):return p
        class Base:
            def _create_sim(self):return self.gym.create_sim(0,-1,'physx',params)
        class Combined(CapacitySafeConstructionMixin,Base):pass
        env=Combined();original=Gym();env.gym=original
        result=env._create_sim()
        self.assertEqual(result.physx.max_gpu_contact_pairs,2**25)
        self.assertEqual(result.physx.default_buffer_size_multiplier,5.)
        self.assertEqual(result.physx.contact_offset,.01)
        self.assertIs(env.gym,original)

    def test_unique_construction_origins(self):
        a=separated_construction_origins(4096)
        self.assertEqual(len(np.unique(a,axis=0)),4096)
        self.assertTrue(np.all(a[:,2]==0))
        self.assertTrue(np.all(a%8==0))

    def test_original_origins_restored_before_base_allocation(self):
        expected=np.array([[4.,4.,0.],[4.,4.,0.]],dtype=np.float32)
        class Base:
            num_envs=2
            def _make_origins(self):return expected
            def _allocate_buffers(self):self.seen=self.env_origins_np.copy()
        class Combined(ConstructionStagingMixin,Base):pass
        env=Combined();env.env_origins_np=env._make_origins()
        self.assertFalse(np.array_equal(env.env_origins_np,expected))
        env._allocate_buffers()
        self.assertTrue(np.array_equal(env.seen,expected))
        self.assertTrue(np.array_equal(expected,[[4,4,0],[4,4,0]]))


if __name__=='__main__':unittest.main()
