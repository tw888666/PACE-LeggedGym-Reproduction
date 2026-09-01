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
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli torque_semantics
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli fit_order_diagnostics
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli bias_law_counterfactual
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli plant_audit
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli accumulation_audit
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

`torque_semantics` is also dataset-only. It enumerates the frozen 18 bias/frame
candidates, pre/post-delay timing, torque-only lag, P/PD, DCMotor clipping, and
velocity/Coulomb residual signatures. It never runs Isaac Gym or changes the formal
actuator/replay implementation.

`fit_order_diagnostics` is a read-only audit of the 12-value parameter-block order in
the frozen `fitting.npy`. It compares both `LF LH RF RH` and `LF RF LH RH` hypotheses
against every joint-labelled ANYmal value in paper Table 6, using publication-rounding
intervals. It also inventories every payload key and searches only the authorized
`pace_data/**/anymal*/fitting.npy` scope. It does not change the decoder or run a
counterfactual replay unless the direct-order hypothesis wins all four parameter blocks.

`bias_law_counterfactual` runs one diagnostic-only Isaac Gym A/B. The public branch uses
`q_control=q_true-bias`; the legacy-effective branch uses `q_control=q_true`. Both retain
the frozen `q_compare=q_true-bias`, initialization, absolute targets, DCMotor envelope,
three-step FIFO, asset, and PhysX settings. The command never changes the default
`PACEActuatorCore` implementation or the formal Stage 0C status.

`plant_audit` keeps Branch A closed and resets the current public-asset simulator to the
author state before every 400 Hz interval. It compares one-step position, velocity, and
acceleration transitions under the frozen public teacher torque and the secondary logged
torque, snapshots all runtime mass/COM/inertia properties, verifies zero contact, then runs
only the mandatory `use_physx_armature` and `collapse_fixed_joints` OFAT diagnostics. It
does not select a new formal baseline or run locomotion/PPO.

`accumulation_audit` bypasses the formal loader for a read-only source-dtype/time-grid
inventory, evaluates position-derived velocity hypotheses at lag -3..+3, and runs a frozen
multi-environment diagnostic matrix covering four initial velocities, target timing ±1,
teacher-forcing horizons 1..512/full, and q-only/qdot-only resets. It preserves the formal
actuator, delay FIFO, initial state, target timing, plant parameters, and Stage 0C gates.
