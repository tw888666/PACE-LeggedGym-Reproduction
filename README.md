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
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli replay_fit
```

The replay preflight reports both `RMSE(sim_method, real)` and
`RMSE(sim_method-bias, real)`. If the frozen frame expectation fails, replay stops and
writes `preflight.json`. After manual review, continuation without changing any frame is
explicitly available as:

```bash
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli replay_fit \
  --acknowledge-frame-sanity-failure
```

This flag never chooses a lower-RMSE frame and does not alter formal metrics.

## Sampling semantics

```text
state[0]       = initialized state before simulation
target[t]      = command over [t,t+1)
state[t + 1]   = state after that simulation interval
```

Formal simulator replication always compares `q_true_ours[k]` with
`sim_method.dof_pos[k]` at zero lag. Lag `-4..+4` is diagnostic only.

