# PACE ANYmal D — public-information independent reproduction

## Project status

```text
Stage 0A       PASS
Stage 0B       FROZEN
Stage 0C-E     FAIL / unresolved
Stage 0C-R     PASS
Stage 0        FROZEN
Stage 1        PIPELINE PASS / TRAINING USABLE
Policy quality NOT FROZEN
PPO training   NOT STARTED IN THIS COMMIT
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
report must retain that banner. Stage 1 locomotion implementation is authorized, but it
has not been started in this freeze commit. PPO implementation, GPU jobs, and PPO training
have not started.

## Stage 1 PACE-derived energy-off task ablation（PACE 派生关闭能耗项任务消融）

新的 Stage 1 正式分支为 `codex/stage1-energy-off-ablation`，诊断分支为
`codex/stage1-energy-off-diagnostic`；二者均从
`stage0-public-reproduction-ready` 直接创建。历史节点 `stage1-ppo-ready` / `c8454db`
保持不变，标记为 `PACE energy MDP v1 Phase A failure`，不作为本分支的代码起点。

该实验是从 PACE v2 四项 locomotion objective（运动目标）中删除 energy term（能耗项）
后得到的项目内消融，不存在于 PACE 论文或官方 release（发布版本）中。因此不得称为
“PACE Task-only baseline（PACE 仅任务基线）”，也不得作为官方论文基线引用。

> This branch implements a PACE-derived energy-off ablation. It is not an official PACE
> baseline because the original release does not provide this experiment.

本阶段仅验证“冻结 Stage 0 physics reconstruction（物理重建）后，ANYmal locomotion
task MDP（运动任务马尔可夫决策过程）是否闭合”。Actor observation（策略观测）为
48 维，critic observation（价值网络观测）为 353 维；12 维 action（动作）先按
LeggedGym 语义裁剪到 `[-100,100]`，再通过 `q_target=q0+0.5*action` 形成绝对关节目标，
并进入未修改的 Stage 0 `PACEActuatorCore`。Stage 1 仍计算带 5-degree（5 度）内缩带的
URDF joint-limit clamp（关节限位裁剪）候选，但默认
`enforce_joint_limit_on_policy_target=false`，该 reconstruction safety wrapper
（重建安全包装层）不参与实际 actuator 输入。这个直接 target clamp 不等价于 PACE v2
Eq. 9；当前边界是“PACE actuator core reproduction without Eq. 9 hard-limit-safe PD
extension（PACE 执行器核心复现，不含 Eq. 9 硬限位安全 PD 扩展）”。

Reward（奖励）只有以下执行项：

- velocity tracking（速度跟踪），scale（系数）`0.2`；
- collision（碰撞），scale `-1.0`；
- FTD / foot touchdown（足端触地），scale `-0.1`，保留 500 iteration（迭代）半衰期；
- termination bookkeeping（终止记账），继承 scale `-0.0`。

完整执行顺序继承 LeggedGym（腿式机器人训练框架）：所有非终止项乘
`policy_dt=0.01 s` 后求和，`only_positive_rewards=true` 时先裁剪到非负，再追加仅对
非 timeout（超时）有效的 termination reward（终止奖励）。Reward 在 terminal
pre-reset state（终止前重置状态）上计算，随后 reset；timeout 通过
`rsl_rl v1.0.2` 的 `gamma*V` 规则自举。该顺序避免“发生终止但奖励无法解释”。

Stage 1 没有 energy reward（能耗奖励）、energy penalty（能耗惩罚）、energy
curriculum（能耗课程）、cost critic（代价价值网络）、Lagrangian multiplier
（拉格朗日乘子）或 budget constraint（预算约束）。权威字段与来源分类见
[`provenance/stage1_task_only_manifest.json`](provenance/stage1_task_only_manifest.json)。

验证命令均为单行：

```bash
conda run --no-capture-output -n bruce_gym env PYTHONPATH=src python -m unittest discover -s tests -v
conda run --no-capture-output -n bruce_gym env PYTHONPATH=src python -m pace_stage1.smoke --num-envs 2 --steps 3 --seed 123
conda run --no-capture-output -n bruce_gym env PYTHONPATH=src python -m pace_stage1.smoke --num-envs 1 --steps 1 --seed 123 --terrain-mode trimesh
conda run --no-capture-output -n bruce_gym env PYTHONPATH=src python -m pace_stage1.ppo_smoke
```

测试结果为 `101/101 PASS`：Stage 0 保留 `71/71`，Stage 1 为 `30/30`。Plane
（平面）和 production trimesh（生产三角网格）环境 smoke（冒烟验证）通过；一次
2-env、2-step 的内存内 PPO 更新也通过，loss（损失）有限，且未创建 checkpoint。
这些 bounded validation（有界验证）不生成正式结果。历史 3000-iteration 运行已经完成，
但仅归档为 diagnostic evidence（诊断证据），其 checkpoint 不具备 baseline 身份。

训练日志在不改变 MDP 的前提下额外记录 completed episode（已完成回合）的
`base_contact_rate`、`timeout_rate`、`velocity_tracking_rmse`、`yaw_tracking_rmse`、
`mean_abs_action`、`torque_saturation_ratio` 和 `mean_torque_utilization`；后两项分别按
`abs(raw_pd_torque-saturated_torque)>1e-6 Nm` 与 `abs(raw_pd_torque)/89 Nm` 定义，并在
4 个 physics substep（物理子步）和 12 个 DOF 上聚合。Policy distribution std
（策略分布标准差）继续使用 rsl_rl 原生
`Policy/mean_noise_std`。新的 energy-off ablation 固定入口为：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n bruce_gym env PYTHONPATH=src python -m pace_stage1.train --flat-seed0
```

该入口固定为 `4096 env × 3000 iterations × seed 0`，使用 plane terrain（平面地形），
保留 friction randomization（摩擦随机化）和 pushes（推扰），新实验名为
`stage1_pace_energy_off_flat_seed0`。历史 `stage1_task_only_flat_seed0/model_3000` 只作为
torque-saturation diagnosis（力矩饱和诊断）保留，不得解释为 baseline。正式 seed0 已从
`stage1-energy-off-semantics-frozen` tag 对应 commit 启动；运行状态以 timestamped
`run_manifest.json` 为准。

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
