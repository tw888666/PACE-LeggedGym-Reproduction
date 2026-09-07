# Stage1 PACE v2 action 证据审计

审计日期：2026-09-07

## 结论

截至审计日期，在当前可访问的论文源文件、官方公开代码及其历史、官方 issue（问题单）
回复和 `pace_data` 数据归档中，没有找到以下两项的权威 locomotion（运动策略）实现值：

1. ANYmal locomotion policy 使用的 12 维默认姿态 $q_0$；
2. rsl_rl actor 的 Gaussian sample（高斯采样）$z_t$ 到论文变量
   $a_t\,[\mathrm{rad}]$ 的完整映射，包括 `tanh`、scale（尺度）、clip（裁剪）或其他变换。

因此审计结果是“已完成规定范围搜索，未找到权威 locomotion 数值”，而不是证明这些实现
细节不存在。三项状态保持：

```text
control.network_output_to_action_offset_mapping    UNRESOLVED
control.action_output_clipping                     UNRESOLVED
control.default_posture_q0                         UNRESOLVED
```

本次审计不改变训练门禁，不接入 Eq. (9)，也不把任何候选值升级为 `PAPER_EXACT`。

## 论文层已经确定的边界

PACE v2 Sec. 2.3.2, Eq. (8) 明确给出：$a_t$ 是相对默认姿态 $q_0$ 的关节位置偏移，
目标为 $\hat q_t=q_0+a_t$。Eq. (9) 后的澄清又明确说明 joint-limit handling
（关节限位处理）不约束 policy action space（策略动作空间），也不阻止 target（目标）越过
hard limits（硬限位）。

所以论文解决的是后半段物理语义，而不是以下实现链的前半段：

```text
z_t (rsl_rl Gaussian sample)
  -> tanh? / scale? / clip? / other transform?
  -> a_t [rad]
  -> +q0
  -> q_target
  -> Eq. (9)
```

PACE v1 源文件已经包含同样的 offset（偏移）语义；v2 增加了 Eq. (9) 及其动作空间
澄清，但两个版本都没有列出 ANYmal 的 12 维 $q_0$，也没有公开上述 network-to-action
映射。

## 官方代码与历史审计

审计固定在公开仓库 commit（提交）
`f07259c09b517ab5118bb1d01b0a6078cf8e1c31`，并检查了可达的 67 个 commit、公开
branch（分支）和 `v0.1.0` 至 `v0.1.2` tag（标签）。结果如下：

- 当前 `JointPositionActionCfg(scale=1.0, use_default_offset=False)` 明确是 absolute joint
  position target（绝对关节位置目标），位于 SysID（系统辨识）/数据采集环境；
- 历史上该 joint-position action（关节位置动作）也是由 SysID 路径引入，后续设置
  `use_default_offset=False` 是为避免部分机器人零位不可行；
- 未发现论文 locomotion training environment（运动训练环境）、deployment config
  （部署配置）、训练 checkpoint（检查点）、ONNX/JIT policy artifact（策略制品）或被删除的
  locomotion action 配置；
- 通用的 rsl_rl `play.py` 导出能力不包含具体 policy，也不能反推出 $q_0$ 或映射。

因此公开 SysID 配置只能作为 `PUBLIC_CODE_CONTEXT_ONLY`，不能证明 PACE locomotion 的
`action_scale=1.0`。

## 官方回复提供的边界

- 官方 issue #30 中，维护者明确说明完整 locomotion 环境不在当前仓库，并表示没有发布
  trained locomotion checkpoints（已训练运动策略检查点）的计划。这解释了为何当前 release
  无法通过训练或部署 artifact 闭合两条证据链。
- issue #22 的维护者回复要求 identification、RL 和 deployment（辨识、强化学习和部署）间
  的 action scale、PD gains、control decimation 和 torque limits 保持一致，但没有给出论文
  ANYmal locomotion 的具体 action scale、clip、$q_0$ 或网络输出变换。
- issue #15 讨论的 Isaac Gym / LeggedGym 移植尚未进入官方 release；讨论中也没有公开上述
  两项数值。
- issue #23 只解释替换 actuator model（执行器模型）后需要重新审视训练时长和任务调参，
  不提供 action mapping 证据。

## 数据归档审计

公开 `pace_data` 的本地 ANYmal 归档包含 `des_dof_pos`、`dof_pos`、`dof_vel`、torque
（力矩）和 base state（基座状态）等时序，但没有 raw actor output（原始策略输出）、$q_0$、
scale/clip metadata（元数据）或可执行 policy artifact。四个已检查归档的 SHA-256 为：

```text
1_nothing.npy  e565c3b868819e1dac96a2f8836f716dc0119bc7d027205e96578ae1fc053e92
2_act_net.npy  a62a37972aab1d8095798e5c1a32cab27546974cbd0ffe6ac3e7ca12b27bb379
3_method.npy   1d6dddb0c0ca623561d06411199bc631f44ff8767f728f44b584e59f6b8e6f8a
additional     89f83a376e295ebf86a3991becc6b33808325d7d9426a540a2d20e129350f031
```

仅有 `des_dof_pos` 无法唯一分解为 $q_0+a_t$，因此不能通过均值、站立片段或其他统计方式
把数据反推结果标成论文精确值。

## 明确排除的替代证据

以下值可以成为未来 reproduction assumption（复现假设）的候选，但当前不能替代论文
locomotion policy 的 $q_0$：

- URDF 零位；
- LeggedGym 默认 ANYmal pose（姿态）；
- SysID 初始姿态或 chirp trajectory bias（扫频轨迹偏置）；
- 当前 Stage1 使用的 `[0, ±0.4, ±0.8]` LeggedGym-derived 姿态。

同理，Eq. (8) 中 $a_t$ 已经具有 rad 单位，不能据此断言 rsl_rl 的 raw Gaussian sample
直接等于 $a_t$，也不能据此把 `action_scale` 改为 1。

## 门禁结果与下一步

```text
action evidence audit         COMPLETE
authoritative q0              NOT FOUND / UNRESOLVED
authoritative z_t -> a_t      NOT FOUND / UNRESOLVED
Eq. (9) integration           DISABLED
PACE_V2_VALIDATION            PROHIBITED
PACE_V2_FORMAL                PROHIBITED
```

下一步只有两条严谨路径：等待/询问作者提供 locomotion 实现细节，或单独形成不可按性能调参的
预注册复现假设。采用后一条路径时，来源必须保持 `REPRODUCTION_ASSUMPTION`，不得改写为
`PAPER_EXACT`，并需在 action mapping 敏感性实验中报告选择影响。

## 可复核来源

- [PACE v2 arXiv HTML](https://arxiv.org/html/2509.06342v2)
- [公开 SysID action 配置](https://github.com/leggedrobotics/pace-sim2real/blob/f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/pace_sim2real_env_cfg.py#L59-L63)
- [官方 issue #30 维护者回复](https://github.com/leggedrobotics/pace-sim2real/issues/30#issuecomment-5140694339)
- [官方 issue #22 维护者回复](https://github.com/leggedrobotics/pace-sim2real/issues/22#issuecomment-4661107618)
- [Isaac Gym 移植讨论 issue #15](https://github.com/leggedrobotics/pace-sim2real/issues/15)
- [`pace_data` DOI](https://doi.org/10.3929/ethz-c-000783505)
