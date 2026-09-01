# PACE ANYmal D — public-information independent reproduction

## Project status

```text
Stage 0A       PASS
Stage 0B       FROZEN
Stage 0C-E     FAIL / unresolved
Stage 0C-R     PASS
Stage 0        FROZEN
Stage 1        IMPLEMENTED / PPO READY
PPO training   NOT STARTED
```

`Stage 0C-E` is the exact legacy-trajectory replay gate. Its original thresholds remain
`overall <= 0.010 rad` and `all per-joint <= 0.020 rad`; the released replay residual is
`0.012818364796375577 rad`, so this gate remains permanently **FAIL / unresolved**.
`Stage 0C-R` is a separate public-information reproduction-readiness decision. It is
**PASS** by explicit human approval of Option C and does not imply exact reproduction of
the unpublished legacy simulator, asset, exporter, or PhysX configuration.

The authoritative machine-readable decision is
[`provenance/stage0_freeze.json`](provenance/stage0_freeze.json). Its final freeze commit
is resolved by the peeled target of the annotated tag
`stage0-public-reproduction-ready`; this avoids recording a fabricated self-referential
commit SHA inside the commit itself.

## Mandatory Stage 1+ provenance banner

```text
Reproduction status:
Public-information independent reproduction.

Exact PACE legacy SysID replay:
UNRESOLVED / Stage 0C-E FAIL.

Known residual:
overall 0.012818 rad on released legacy replay.

Legacy exact asset and paper-era simulator configuration:
not publicly resolved.
```

Every future Stage 1+ README section, experiment manifest, training report, and evaluation
report must retain that banner. Stage 1 locomotion environment/spec implementation now
lives on the branch created from this freeze. No formal PPO rollout, GPU job, training
checkpoint, or research result has been started or created.

## Stage 1 locomotion environment (pre-PPO)

```text
Reproduction status:
Public-information independent reproduction.

Exact PACE legacy SysID replay:
UNRESOLVED / Stage 0C-E FAIL.

Known residual:
overall 0.012818 rad on released legacy replay.

Legacy exact asset and paper-era simulator configuration:
not publicly resolved.
```

Stage 1 follows `LeggedGym standard locomotion + PACE explicit changes + minimal
reconstruction`. The authoritative fixed values and per-field evidence classifications are
in [`provenance/stage1_manifest.json`](provenance/stage1_manifest.json); executable values
are in [`src/pace_stage1/config.py`](src/pace_stage1/config.py).

The actor observation is 48-dimensional in this order: body-frame base linear velocity,
body-frame base angular velocity, projected gravity, `vx/vy/yaw-rate` command, encoder
joint position, joint velocity, and previous action. The asymmetric critic input is 353
dimensional: the noise-free 48-vector plus base wrench, friction, four foot contacts, and a
294-value base-centered height scan. Actions are 12 canonical joint offsets with scale
`0.5 rad`; the frozen locomotion target adapter emits absolute targets into the unchanged
Stage 0 PACE actuator, whose 3-step torque FIFO advances at 400 Hz. The policy target is
held for four physics steps, so the policy rate is 100 Hz.

The four rewards are PACE velocity tracking (`0.2`), energy (`-16e-5`), collision (`-1`),
and 3-sample foot-touchdown speed (`-0.1`). Energy and FTD use the public exponential
500-iteration penalty ramp. The unreleased ANYmal electrical coefficient is fixed to an
explicit low-confidence reconstruction value; it is not source-confirmed. Terrain is the
inherited 10x20 LeggedGym mixed-terrain curriculum; friction and pushes are randomized,
while dynamics, base mass, and motor strength are not randomized.

The bounded pre-freeze environment smoke entry point is:

```bash
conda run -n bruce_gym env PYTHONPATH=src \
  python -m pace_stage1.smoke --num-envs 2 --steps 3 --seed 123
```

The smoke command enforces `num_envs <= 8` and `steps <= 16`, labels itself
`SMOKE / NON-EXPERIMENTAL`, does not import rsl_rl, and cannot create a checkpoint.

The frozen semantic suite is `91/91 PASS` (the original Stage 0 `71/71` plus 20
Stage 1 tests). Both plane and production-trimesh CPU construction/step smoke pass,
including repeated same-seed tensor-hash equality. A final GPU0 production-trimesh
environment-only check also passed with 16 environments for three policy steps; it did
not construct a runner, start PPO, or create a checkpoint. Therefore the audited pre-training
decision is `Stage 1 PPO gate: PASS`; this decision authorizes a later, separate formal
baseline-training stage but does not itself start PPO.

The final freeze authorization is
[`provenance/stage1_ppo_authorization.json`](provenance/stage1_ppo_authorization.json).
The only authorized first run is `PHASE A / PIPELINE VALIDATION / NON-PAPER` with
`1024 env × 300 iterations × seed 1` on `cuda:0`, named
`stage1_smoke_ppo_seed1`. It must be launched from the peeled target of the annotated
tag `stage1-ppo-ready`; the launcher rejects a moved Stage 0 tag, dirty worktree,
wrong rsl_rl version, unavailable GPU0, or any non-Phase-A invocation.

The precise energy-model interpretation boundary is: *the energy term follows the
publicly available formulation, while unavailable electrical conversion parameters are
fixed according to the reconstruction manifest.* It must not be described as an exact
reproduction of the PACE energy model.

## Frozen Stage 0 scope

The Stage 0 implementation is frozen at this boundary:

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
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli p4_p5_audit
conda run -n bruce_gym env PYTHONPATH=src python -m pace_stage0.cli boundary_review
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

`p4_p5_audit` is the final narrow Stage 0C OFAT diagnostic authorized by the accumulation
audit. P4 compares only fitted joint properties against `damping=0`, `friction=0`, and
both-zero variants under one-step teacher forcing plus H8/H32/H128/full horizons. Isaac
Gym joint friction is treated as a dimensionless, transmission-force/load-dependent
Coulomb coefficient—not a fixed Nm torque—following the
[NVIDIA engineer explanation](https://forums.developer.nvidia.com/t/possible-bug-in-joint-friction-value-definition/208631).
Only if P4 has no evidence-backed material improvement does P5 run exactly `substeps=2`
and `num_velocity_iterations=1`. The command never scales or refits parameters, never
uses variable timestamp dt, and never promotes a lower-RMSE diagnostic to the formal
baseline.

`boundary_review` is a static pre-approval evidence snapshot: it does not launch Isaac
Gym. It combines the frozen Stage 0 reports into separate tracks for exact legacy replay
(`Stage 0C-E`) and public-information reproduction readiness (`Stage 0C-R`). That report
retains `Stage 0C-E = FAIL/unresolved`, recommends Option C, and intentionally records
`Stage 0C-R = NOT SET` because it cannot infer a human decision. The subsequent explicit
human approval and current `Stage 0C-R = PASS` are recorded only in
`provenance/stage0_freeze.json`. The exact thresholds and failure evidence remain unchanged.
