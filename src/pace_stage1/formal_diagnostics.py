"""Read-only formal-run diagnostics. Thresholds are raw actor output units."""
import torch


def action_tail_summary(samples):
    values = torch.cat(samples, dim=0).abs()
    flat = values.flatten()
    quantiles = torch.quantile(flat, torch.tensor([.95,.99,.999,.9999], device=flat.device))
    result = dict(zip(('p95','p99','p999','p9999'), quantiles.cpu().tolist()))
    result['scalar_count'] = flat.numel()
    for threshold in (2,5,10):
        mask = (values > threshold).float()
        result['prob_abs_gt_%d' % threshold] = mask.mean().item()
        result['per_joint_prob_abs_gt_%d' % threshold] = mask.mean(0).cpu().tolist()
    return result


class FormalDiagnosticsMixin:
    def reset_metrics(self):
        super().reset_metrics()
        self.raw_samples = []
        self.joint_saturation = torch.zeros(12, device=self.device)
        self.energy_sum = torch.zeros(3, device=self.device)
        self.target_min = torch.full((12,), float('inf'), device=self.device)
        self.target_max = -self.target_min.clone()
        self.tail_context = torch.zeros(8, device=self.device)

    def step(self, actions):
        raw = actions.detach().clone()
        self.raw_samples.append(raw)
        self._raw_for_metrics = raw
        tail = raw.abs() > 10
        reset_near = self.episode_length_buf < 10
        speed = self.commands[:, :2].norm(dim=1)
        for offset, mask in ((0, reset_near), (4, speed < .2), (6, speed >= .2)):
            self.tail_context[offset] += tail[mask].sum()
            self.tail_context[offset+1] += mask.sum()*12
        result = super().step(actions)
        done = result[3].bool()
        self.tail_context[2] += tail[done].sum()
        self.tail_context[3] += done.sum()*12
        return result

    def _step_actuator(self, target):
        result = super()._step_actuator(target)
        saturation = (result.raw_pd_torque-result.saturated_torque).abs() > 1e-6
        self.joint_saturation += saturation.float().mean(0)
        self.target_min = torch.minimum(self.target_min, target.amin(0))
        self.target_max = torch.maximum(self.target_max, target.amax(0))
        electrical = .0192*result.applied_torque.square().sum(1)
        signed = (result.applied_torque*self.dof_vel).sum(1)
        self.energy_sum += torch.stack((electrical.mean(), signed.mean(), signed.clamp_min(0).mean()))
        return result

    def metrics(self):
        result = super().metrics()
        result['raw_action_tail'] = action_tail_summary(self.raw_samples)
        result['per_joint_torque_saturation'] = (self.joint_saturation/self.audit_substeps).cpu().tolist()
        result['q_target_raw_min_rad'] = self.target_min.cpu().tolist()
        result['q_target_raw_max_rad'] = self.target_max.cpu().tolist()
        power = (self.energy_sum/self.audit_substeps).cpu().tolist()
        result['energy_logging_only'] = dict(electrical_w=power[0], mechanical_signed_w=power[1],
                                            mechanical_positive_net_w=power[2], electrical_plus_positive_net_w=power[0]+power[2])
        result['tail_gt10_context'] = {name: {'tail_scalars': self.tail_context[2*i].item(),
                                             'all_scalars': self.tail_context[2*i+1].item()}
                                     for i,name in enumerate(('first10_steps_after_reset','termination_step','command_speed_lt_point2','command_speed_ge_point2'))}
        return result


def normalization_summary(normalizers):
    result = {}
    for name in ('actor','critic'):
        norm = getattr(normalizers, name)
        for value in norm.state_dict().values():
            if not torch.isfinite(value).all():
                raise RuntimeError('non-finite normalizer state')
        if torch.any(norm._var < 0) or torch.any(norm._std < 0):
            raise RuntimeError('negative normalizer variance/std')
        result[name] = {'count': norm.count.item(), 'variance': norm._var.flatten().cpu().tolist()}
    return result
