# Stage1 PACE v2 阻塞项重分类

审计日期：2026-09-07

## 结论

原始 12 个 formal blocker（正式训练阻塞项）中，动作、公式（9）接线/带宽及观测归一化决议已解决八个子项，奖励语义决议又解决三个，当前剩余 1 项分类如下：

```text
semantic_blockers              = 0
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

## 奖励项后续解决

[奖励语义预注册 v1](gpt-Stage1-PACEv2-奖励语义预注册-v1.md)及组件验证已完成。
当前登记的语义项清零，只剩 `ppo.entropy_slope_eta`。短程验证仍须单独确定斜率和完整入口，
门禁不自动开放。下文保留各次重分类的历史说明。

## 观测项后续拆分

[观测归一化只读审计](gpt-Stage1-PACEv2-观测归一化只读审计.md)将当前观测阻塞项
明确分为 `observation.component_scaling`、`observation.noise_and_scaling_order` 和
`observation.empirical_normalization_semantics`。下表保留初始分类；当前标识以矩阵为准。
随后[归一化预注册及实现](gpt-Stage1-PACEv2-观测归一化重建预注册-v1.md)已解决这三个观测项。
当前仅剩奖励侧 3 个语义项及熵斜率 1 个可复现性项，门禁继续关闭。

## 原始 12 项重分类及当前处理状态

优先级 1 的三项现已由[动作预注册决议 v1](gpt-Stage1-PACEv2-动作重建预注册决议-v1.md)
解决，状态均为 `RESOLVED_BY_PREREGISTERED_ASSUMPTION`，不再阻塞训练。优先级 2 的接线与 5° 带宽也已由
[公式（9）决议](gpt-Stage1-PACEv2-公式9接线与软带宽预注册-v1.md)解决；观测三个项也已解决；其余 4 项继续阻塞。

| 优先级 | blocker | 首要分类 | 为什么会阻塞 | 允许的解决方式 |
|---:|---|---|---|---|
| 1 | `control.network_output_to_action_offset_mapping` | SEMANTIC | scale、tanh 或其他 transform 会改变 raw policy output 到论文变量 `a_t` 的映射 | validation 前冻结唯一协议值 |
| 1 | `control.action_output_clipping` | SEMANTIC | environment-side clip 会改变 policy action space 的有效边界 | validation 前冻结唯一协议语义 |
| 1 | `control.default_posture_q0` | SEMANTIC | 12 维数值改变所有关节 target 的零点 | validation 前冻结唯一协议值 |
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

## Action representation 的已解决与未决边界

PACE v2 Eq. (8) 已明确 action 是相对默认姿态的 joint-position offset（关节位置偏移）。
因此以下两项已经是 `PAPER_EXACT / MATCH`，不属于 blocker：

```text
action_physical_semantics:
    a_t = relative joint-position offset [rad]

target_mapping:
    q_target = q0 + a_t
```

动作预注册前未决的不是“位置控制还是力矩控制”，而是以下实现层组合的完整数值语义：

```text
raw policy output
    -> environment-side action clip
    -> action scale
    -> q0 + scaled action
    -> Eq. (9) hard-limit-safe target reshape
    -> PD controller
```

优先级 1 的三个子项现已预注册；Eq. (9) 的最终接线和 5° 带宽也已登记并接入独立验证路径。

## 公开 pace-sim2real 配置的证据边界

公开仓库 `f07259c09b517ab5118bb1d01b0a6078cf8e1c31` 中的配置是：

```text
JointPositionActionCfg(
    scale=1.0,
    use_default_offset=False
)
action semantics = absolute joint position targets
```

该配置属于 system-identification/data-collection（系统辨识/数据采集）路径，不是论文
locomotion policy 的完整训练环境。因此它只能证明该公开 SysID 路径的 absolute target
语义，不能证明 PACE locomotion 使用 `action_scale=1.0`。机器可读矩阵将其标记为
`PUBLIC_CODE_CONTEXT_ONLY`，且 `locomotion_evidence=false`。

## 下一步审计顺序

1. 动作重建预注册已完成；保留作者回复渠道，按决议的启动前覆盖、启动后版本分离规则处理新证据。
2. Eq. (9) 接线与 5° 带宽已解决；历史路径保持不变。
3. 观测归一化组合及训练/推理/恢复语义已预注册并实现，组件和跨进程恢复验证通过。
4. 三个奖励语义项已预注册并通过组件验证；当前登记语义项清零。
5. 语义项清零后才显式打开短程 validation；`eta` 可用预注册诊断值做敏感性测试，
   但在正式来源或正式假设冻结前，formal gate（正式门禁）继续关闭。
