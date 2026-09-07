# Stage1 PACE v2 阻塞项重分类

审计日期：2026-09-07

## 结论

当前 12 个 formal blocker（正式训练阻塞项）按其首要影响分类如下：

```text
semantic_blockers              = 11
reproducibility_blockers       = 1
formal_reporting_blockers      = 0
validation_training_allowed    = false
formal_training_allowed        = false
```

这里不强行采用示例性的 `5/4/3` 数量。action（动作）、transition（状态转移）、
observation（观测）和 reward（奖励）任一语义变化都会改变训练问题本身，因此不能降级为
“只影响报告”。Entropy slope `eta`（熵调度斜率）不改变 MDP（马尔可夫决策过程），但会
改变优化轨迹，所以归入 reproducibility blocker（可复现性阻塞项）。

## 两层训练门禁

```text
纯函数/组件测试
    始终允许

PACE_V2_VALIDATION 短程训练
    semantic_blockers == 0
    且 validation_training_allowed == true

PACE_V2_FORMAL 正式训练
    所有 blocker == 0
    且 formal_training_allowed == true
```

当前只允许纯函数、配置和 checkpoint（检查点）保存恢复测试，不允许任何
`PACE_V2_VALIDATION` 训练，更不允许 30,000 iterations（迭代）的正式训练。

## 12 项重分类和处理顺序

| 优先级 | blocker | 首要分类 | 为什么会阻塞 | 允许的解决方式 |
|---:|---|---|---|---|
| 1 | `control.action_scale_rad` | SEMANTIC | 改变 raw policy output 到关节位置 offset 的映射 | validation 前冻结唯一协议值 |
| 1 | `control.action_clip` | SEMANTIC | 改变 policy action space 的有效边界 | validation 前冻结唯一协议语义 |
| 1 | `control.default_joint_pose_q0` | SEMANTIC | 改变所有关节 target 的零点 | validation 前冻结唯一协议值 |
| 2 | `control.eq9_joint_limit_saturation` | SEMANTIC | 是否接入及接入位置会改变施加给 PD 的 target | validation 前冻结唯一接线语义 |
| 2 | `control.soft_limit_band_deg` | SEMANTIC | 改变 Eq. (9) 生效区间和 transition | 在论文 `2–5°` 范围内预注册复现选择；禁止按性能选值 |
| 3 | `observation.fixed_component_scales` | SEMANTIC | 改变 policy 输入坐标系 | 与 normalization composition 一并冻结 |
| 3 | `observation.empirical_normalization` | SEMANTIC | 接入后改变 actor/critic 的输入分布 | validation 前冻结更新、推理和 checkpoint 恢复语义 |
| 3 | `observation.fixed_scale_plus_empirical_normalization_composition` | SEMANTIC | 决定 running moments 统计 raw obs 还是 fixed-scaled obs | validation 前唯一确定处理顺序 |
| 3 | `reward.velocity_tracking_sigma` | SEMANTIC | 改变速度误差奖励曲线，即改变 reward function | 允许透明预注册 LeggedGym-derived 复现值 |
| 3 | `reward.only_positive_aggregation` | SEMANTIC | 改变逐步总奖励的聚合结果 | 允许透明预注册开关，但 validation 前必须冻结 |
| 3 | `reward.scale_by_policy_dt` | SEMANTIC | 改变奖励权重的离散时间解释 | 允许透明预注册语义，但 validation 前必须冻结 |
| 4 | `ppo.entropy_slope_eta` | REPRODUCIBILITY | 不改变 MDP，但改变 exploration（探索）和 PPO 优化轨迹 | 预注册假设；正式报告做 sensitivity（敏感性）分析 |

`FORMAL_REPORTING` 当前为空：12 项中没有任何一项仅靠在论文中补一句说明就能消除；
每一项都会影响实际控制、输入、奖励或优化过程。

## Action representation 的边界

PACE v2 Eq. (8) 已明确 action 是相对默认姿态的 joint-position offset（关节位置偏移）。
当前未决的不是“位置控制还是力矩控制”，而是以下组合的完整数值语义：

```text
raw policy output
    -> environment-side action clip
    -> action scale
    -> q0 + scaled action
    -> Eq. (9) hard-limit-safe target reshape
    -> PD controller
```

因此优先级 1 的三个子项没有冻结前，不应决定 Eq. (9) 的最终接线配置。

## 下一步审计顺序

1. 对照 PACE 作者实现、补充材料和固定 LeggedGym commit，追溯 action scale、clip 和
   ANYmal `q0`；找不到精确来源时，形成一次性预注册选择，不做 locomotion 调参。
2. 冻结 Eq. (9) 接线位置，并把 `5°` 或其他论文范围内取值登记为预注册复现选择。
3. 唯一确定 observation normalization（观测归一化）的组合顺序及训练/推理/checkpoint
   语义。
4. 在 validation 前同时冻结三个 reward 语义项；它们不能留到“报告阶段”。
5. 语义项清零后才显式打开短程 validation；`eta` 可用预注册诊断值做敏感性测试，
   但在正式来源或正式假设冻结前，formal gate（正式门禁）继续关闭。
