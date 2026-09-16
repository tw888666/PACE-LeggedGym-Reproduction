"""PACE energy components; potential sign must be selected explicitly."""
import torch
from .semantics import foot_touchdown_schedule


def energy_components(applied_torque, joint_velocity, body_mass, body_vertical_velocity,
                      *, potential_sign):
    if potential_sign not in (-1,1):
        raise ValueError('potential sign requires an explicit reconstruction decision')
    electrical=.0192*applied_torque.square().sum(-1)
    mechanical=(applied_torque*joint_velocity).sum(-1).clamp_min(0.)
    potential=potential_sign*9.81*(body_mass*body_vertical_velocity).sum(-1)
    return torch.stack((electrical,mechanical,potential),-1)


def energy_reward(mean_power_components, commands, iteration, policy_dt=.01):
    # Commands contain planar linear speed and yaw; angular speed is not in gamma_v.
    gamma_v=1./(1.+commands[:,:2].square().sum(-1))
    unscaled=gamma_v*mean_power_components.sum(-1)
    weighted=policy_dt*(-16e-5)*foot_touchdown_schedule(iteration,500.)*unscaled
    return weighted,unscaled,gamma_v
