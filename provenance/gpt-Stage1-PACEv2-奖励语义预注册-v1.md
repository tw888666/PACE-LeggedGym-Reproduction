# PACE v2 奖励语义预注册 v1

日期：2026-09-07；标识：`REWARD_SEMANTICS_RECONSTRUCTION_V1`；所属规范：`PACE_V2_SPEC_V1`。
在本轮实现和测试前登记；不以运动性能选值。作者后续证据沿用动作预注册 v1 的覆盖和版本分离规则。

## 三项决议

| 项目 | 主配置 | 来源与选择理由 |
|---|---|---|
| 速度跟踪分母 | `tracking_sigma=0.25` | 固定 LeggedGym（足式运动框架）派生假设；保留当前数值 |
| 总奖励非负裁剪 | `only_positive_rewards=False` | 按论文公式（20）保留带符号加权和的实现解释；不再附加 `max(0,r)` |
| 奖励时间步缩放 | 所有任务项统一乘 `policy_dt=0.01 s` | 固定基础框架的离散时间约定；保留现有量级 |

三项均为 `paper_exact=false`。公式（10）的形状及公式（20）的加权和属于论文明确规范，
但论文未列速度跟踪分母的数值，也未明确描述实现层的非负裁剪开关及额外时间步缩放。
关闭非负裁剪标为 `PAPER_DERIVED_REPRODUCTION_ASSUMPTION`（论文公式派生复现假设），
另两项标为 `LEGGEDGYM_DERIVED_REPRODUCTION_ASSUMPTION`（基础框架派生复现假设）。

[论文第 2.3.3 节](https://arxiv.org/html/2509.06342v2#S2.SS3.SSS3)、
[固定基础框架参数](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot_config.py)、
[基础框架奖励聚合](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py)。

## 可执行的阶段 1 定义

```math
r_v=\exp\left(-\frac{\|\hat v_{xy}-v_{xy}\|^2}{0.25}\right)
+\exp\left(-\frac{(\hat\omega_z-\omega_z)^2}{0.25}\right)
```

这里的 `0.25` 是指数分母本身，不再平方。两个误差分别在 m/s（米每秒）和 rad/s（弧度每秒）
坐标下计算，沿用同一个数值分母约定，不把它解释成通用的物理标准差。

```math
r_t=0.01\left(0.2r_v-r_c-0.1\kappa r_{\mathrm{ftd}}\right),
\qquad \kappa=1-2^{-i/500}.
```

`r_c` 是碰撞或关节越界的布尔指示量；足端触地项沿用三步速度历史和现有 500 次迭代半衰期。
能耗项在阶段 1 中关闭，终止项权重维持 0。不修改碰撞判定阈值、终止顺序或时间截断自举。
例如跟踪完全正确且发生碰撞时，奖励为 `0.01*(0.4-1)=-0.006`，不截断为零。

统一乘时间步在固定步长下保持各任务项相对权重，但会改变价值目标和优化量级；
不能声称对 PPO（近端策略优化）训练轨迹完全无影响。后续阶段 2/3 保持同一任务奖励约定，
新增能耗项/约束的单位转换需在相应阶段明确，不能把已经积分的能量重复乘时间步。

## 实现范围与门禁

通过独立的 PACE-v2 环境配置选择上述三项；历史 `STAGE1_CONFIG`（阶段 1 默认配置）和
原奖励计算函数不改，现有历史训练命令及已有权重不变。未来新路径允许负奖励，训练结果会与历史不同。

验证覆盖分母量级、碰撞负奖励、触地课程、统一时间步因子及历史路径回归，不做训练选参。
通过后只关闭三个奖励阻塞项；正式训练仍受熵斜率阻塞。
语义清零不自动打开短程训练：还需单独确定短程验证的熵斜率及检查完整入口接线，显式开启验证门禁。

## 实现与验证结果

[独立奖励配置](../src/pace_stage1/pace_v2_rewards.py)已接入 PACE-v2 验证环境构造，
复用原任务奖励计算函数，历史默认配置不变。
PACE v2 相关 31 项测试（含本轮新增的 3 项奖励测试）和历史 Stage1 的 30 项回归测试全部通过。
三个奖励项已标记为 `RESOLVED_BY_PREREGISTERED_ASSUMPTION`（通过预注册假设解决）。
当前登记的语义阻塞项为 0，正式阻塞项仅剩 `ppo.entropy_slope_eta`；两层训练门禁仍为关闭。
本轮没有启动仿真训练或修改已有模型权重。
