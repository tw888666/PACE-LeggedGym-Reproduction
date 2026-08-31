# PACE ANYmal D — Stage 0 actuator reproduction

This repository implements only the frozen Stage 0 boundary:

```text
check_install -> decode_fit -> actuator_unit -> replay_fit
```

It does not contain a locomotion environment, rewards, terrain, curriculum, or PPO.

## Frozen control boundary

`replay_fit` sends `real.des_dof_pos[t]` directly to `PACEActuatorCore` as an absolute
joint-position target. It never uses action scaling, a default-pose offset, or the
locomotion hard-limit adapter.

```text
absolute q_target
  -> q_encoder = q_true - bias
  -> PD(Kp=85, Kd=0.6)
  -> four-quadrant DCMotor envelope
  -> saturated-torque 3-step FIFO
  -> Isaac Gym DOF_MODE_EFFORT
```

`LocomotionTargetAdapter` is a separate, unit-tested pure function and is not connected
to Stage 0 replay.

## Runtime

The installed `bruce_gym` Conda environment matches the frozen runtime. Run without
installing extra packages:

```bash
cd /home/xy.chen/tw/PACE-LeggedGym-Reproduction
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli check_install
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli decode_fit
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli actuator_unit
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli frame_forensics
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli replay_fit
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli residual_diagnostics
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli joint_order_diagnostics
```

`frame_forensics` is read-only: it computes H0/H+/H- frame diagnostics and writes
`artifacts/frame_forensics.json` plus `artifacts/frame_forensics.md`. It does not import
Isaac Gym or alter any formal replay metric.

The legacy `data.npy` exporter remains unavailable. Based jointly on the paper, later
official collector/CMA-ES/plotting semantics, official dataset usage, and legacy H0/H+/H-
forensics, Stage 0C freezes this engineering assumption:

```text
sim_method.dof_pos semantic frame = bias-corrected comparison frame
provenance = strong_inference
confidence = high
```

Formal replication therefore compares `q_encoder_ours = q_true_ours - encoder_bias`
against `sim_method.dof_pos`. The frame is fixed before replay and is never selected by
minimum RMSE. Provenance may be upgraded to `author_confirmed` only after explicit author
confirmation.

## Sampling semantics

```text
state[0]       = initialized state before simulation
target[t]      = command over [t,t+1)
state[t + 1]   = state after that simulation interval
```

Formal simulator replication always compares `q_encoder_ours[k]` with
`sim_method.dof_pos[k]` at zero lag. Lag `-4..+4` is diagnostic only.

`residual_diagnostics` does not change the formal replay. It audits legacy fitting bounds,
runs a fixed self-collision OFF/ON A/B, and localizes residuals in torque, velocity, then
position. Its alternate configuration is diagnostic only and cannot select a new baseline.

`joint_order_diagnostics` audits the four frozen RAW/CANONICAL hypotheses, all 24
leg-block bias permutations, the target-position channel signature, and the RF/LH
parameter-assignment risk. It does not modify `decoder.py`, `data.py`, the actuator,
or the formal Stage 0C baseline.
