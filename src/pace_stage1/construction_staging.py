"""Optional construction-only separation; no changes to policy reset origins."""
import numpy as np


def separated_construction_origins(num_envs):
    side=max(1,int(np.ceil(np.sqrt(num_envs))))
    indices=np.arange(num_envs)
    return np.stack((8.*(indices//side),8.*(indices%side),np.zeros(num_envs)),axis=1).astype(np.float32)


class ConstructionStagingMixin:
    """Opt-in only: restore the original layout before initial state allocation."""
    def _make_origins(self):
        self._policy_origins_before_staging=super()._make_origins().copy()
        return separated_construction_origins(self.num_envs)

    def _allocate_buffers(self):
        self.env_origins_np=self._policy_origins_before_staging
        super()._allocate_buffers()


class CapacitySafeConstructionMixin(ConstructionStagingMixin):
    """Candidate opt-in repair, leaving the original formal defaults untouched."""
    def _create_sim(self):
        original=self.gym
        class Proxy:
            def __getattr__(self,name):return getattr(original,name)
            def create_sim(proxy_self,device,graphics,engine,params):
                params.physx.max_gpu_contact_pairs=2**25
                return original.create_sim(device,graphics,engine,params)
        self.gym=Proxy()
        try:return super()._create_sim()
        finally:self.gym=original
