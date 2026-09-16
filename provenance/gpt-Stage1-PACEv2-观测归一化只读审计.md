# Stage1 PACE v2 观测归一化只读审计

审计日期：2026-09-07。只读范围是论文、公开上游及当前运行实现；本次仅更新审计文档和来源矩阵，
不修改归一化器、运行器、环境或训练参数，不进行训练性能比较。

## 后续决议

本文件保留只读审计时的来源缺口。随后已完成[观测归一化重建预注册 v1](gpt-Stage1-PACEv2-观测归一化重建预注册-v1.md)
和兼容层验证，三个观测项通过明确复现假设解决；PACE 论文原训练版本仍未确认。当前状态以来源矩阵为准。

## 结论与三个独立问题

尚不能锁定 PACE 论文运动训练使用的 RSL-RL（强化学习框架）/Isaac Lab（仿真学习框架）版本。
已经核实不同官方框架版本的具体实现，但不能把这些结果直接升级为 PACE 精确实现。

[来源矩阵](gpt-pace-v2-protocol-matrix.json) 现在将观测阻塞项拆成：

| 独立问题 | 矩阵标识 | 本次结论 |
|---|---|---|
| 固定分量尺度 | `observation.component_scaling` | 保留当前基础框架派生复现假设；正式组合未解决 |
| 噪声、尺度与裁剪顺序 | `observation.noise_and_scaling_order` | 当前实现已查清；PACE 原实现未确认 |
| 经验归一化完整语义 | `observation.empirical_normalization_semantics` | 引擎版本、输入、统计更新及恢复协议未冻结 |

`observation.empirical_normalization` 单独保留 Table 8（表 8）的 `True`，其
`PAPER_EXACT`（论文精确规范）只覆盖“启用”，不能覆盖 epsilon（稳定性常数）或统计算法。
原来的尺度/组合标识由上表替代；观测仍有三个聚合阻塞项，不将下面的实现细节重复计数。

## 版本证据能确定到哪里

1. 本项目安装的 `rsl-rl` 是 **1.0.2**，本地来源提交为
   `2ad79cf0caa85b91721abfe358105f869a784121`。它的运行器没有观测经验归一化，
   保存/恢复路径也没有独立归一化统计量。不能用“沿用 v1.0.2”自动推出新引擎。
   [官方 v1.0.2 运行器](https://github.com/leggedrobotics/rsl_rl/blob/2ad79cf0caa85b91721abfe358105f869a784121/rsl_rl/runners/on_policy_runner.py)
2. [论文第 2.3.1 节和表 8](https://arxiv.org/html/2509.06342v2)给出观测内容、维度、加噪对象和启用经验归一化，
   没有给出 RSL-RL 版本或这条观测处理链。重新检索已有 v2 源文件未找到版本锁定。
3. PACE 公开提交 `f07259c09b517ab5118bb1d01b0a6078cf8e1c31` 的
   [安装文档](https://github.com/leggedrobotics/pace-sim2real/blob/f07259c09b517ab5118bb1d01b0a6078cf8e1c31/docs/installation.md)
   写有 Isaac Sim 5.0、Isaac Lab 0.46.2、Python 3.11.13；
   [训练脚本](https://github.com/leggedrobotics/pace-sim2real/blob/f07259c09b517ab5118bb1d01b0a6078cf8e1c31/scripts/rsl_rl/train.py)
   检查的是 **`rsl-rl-lib >= 3.0.1`**，不是精确锁定 3.0.1。
4. 同一提交的[代理配置](https://github.com/leggedrobotics/pace-sim2real/blob/f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/agents/rsl_rl_ppo_cfg.py)
   使用 `cartpole_direct`（倒立摆示例）名称、150 次迭代、`[32,32]` 网络，以及两个关闭的观测归一化开关。
   这与论文设置不同，不能作为运动策略训练配置。
5. Isaac Lab v2.0.0 的运行器配置已经包含 `empirical_normalization` 字段，
   而 RSL-RL v3.0.1 仍能兼容该旧字段；所以字段名称不足以唯一识别版本。
   [Isaac Lab 配置](https://github.com/isaac-sim/IsaacLab/blob/v2.0.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)、
   [RSL-RL v3.0.1 兼容逻辑](https://github.com/leggedrobotics/rsl_rl/blob/2fc1f78bc1d796ffa8f07ce6b09898227db284bb/rsl_rl/runners/on_policy_runner.py)

用户提供的 `ccrpRepo/AMP_mjlab` 页面是第三方仓库，已打开核对其归一化调用；
本审计采用上面的官方版本作依据，不把该第三方页面称为官方 RSL-RL 或 PACE 证据。

对固定 PACE 提交的 `source`、`scripts` 搜索 `obs_scales / observation_scale / noise_scale /
empirical_normalization`，没有发现论文运动环境的对应配置。
公开系统辨识观测组仅含关节位置、速度、上一步动作，且关闭观测扰动；它不是论文的 48/353 维路径。
[公开观测组](https://github.com/leggedrobotics/pace-sim2real/blob/f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/pace_sim2real_env_cfg.py)

扩展历史内容搜索未完成：已有部分克隆缺少历史对象，读取时触发对象补取，而本地代理隧道不可用。
没有重试网络写操作、修改代理或要求用户为本次审计开隧道；固定提交和官方版本的只读源码仍可核实。
因此结论限定为当前已检查范围，不能声称已穷尽全部历史。

## 框架对照：不是 PACE 版本确认

| 项目 | 当前自建组件 | 官方 v2.2.4 | 官方 v3.0.1 |
|---|---|---|---|
| 分母 | `sqrt(var + 1e-8)` | `std + 1e-2` | `std + 1e-2` |
| 归一化输出裁剪 | ±100 | 无 | 无 |
| 更新触发 | 显式 `update=True` | 训练模式前向调用先更新 | `PPO.process_env_step` 显式更新；前向只读 |
| 统计停止阈值 | 无 | 运行器设置 `until=1e8` 样本 | 模型使用默认 `until=None` |
| 策略/价值统计 | 独立 48/353 维 | 独立，运行器持有 | 独立，模型持有 |
| 缓存观测 | 尚未接入运行器 | 缓存归一化后观测 | 缓存环境观测，网络前向时归一化 |
| 统计恢复 | 自建子字典和配置元数据 | 两个独立状态字典 | 随模型状态字典恢复 |

官方对照固定为：

- v2.2.4：`f80d4750fbdfb62cfdb0c362b7063450f427cf35`，
  [归一化模块](https://github.com/leggedrobotics/rsl_rl/blob/f80d4750fbdfb62cfdb0c362b7063450f427cf35/rsl_rl/modules/normalizer.py)、
  [运行器](https://github.com/leggedrobotics/rsl_rl/blob/f80d4750fbdfb62cfdb0c362b7063450f427cf35/rsl_rl/runners/on_policy_runner.py)。
  该运行器在环境步进后归一化；`learn`（学习方法）开始取得的第一批观测没有先经过归一化。
  推理入口切到评估模式。是否保留这个首次观测行为必须明确登记，不能笼统称“每次网络前都相同”。
- v3.0.1：`2fc1f78bc1d796ffa8f07ce6b09898227db284bb`，
  [归一化模块](https://github.com/leggedrobotics/rsl_rl/blob/2fc1f78bc1d796ffa8f07ce6b09898227db284bb/rsl_rl/networks/normalization.py)、
  [策略/价值模型](https://github.com/leggedrobotics/rsl_rl/blob/2fc1f78bc1d796ffa8f07ce6b09898227db284bb/rsl_rl/modules/actor_critic.py)、
  [PPO 算法](https://github.com/leggedrobotics/rsl_rl/blob/2fc1f78bc1d796ffa8f07ce6b09898227db284bb/rsl_rl/algorithms/ppo.py)。
  模块的 `update`（更新方法）在评估模式不更新；初始统计为均值 0、标准差 1、计数 0。

当前官方主分支也已按提交 `00e13d1aa49b398ae512f1765297f7ab8c50ca07` 核对，
[前向计算与更新是分离的](https://github.com/leggedrobotics/rsl_rl/blob/00e13d1aa49b398ae512f1765297f7ab8c50ca07/rsl_rl/modules/normalization.py)。
不能把 v2.2.4 的前向更新规则拼接到当前文件上。

矩阵新增 `empirical_norm_epsilon`、`empirical_norm_output_clip`、`empirical_norm_update_timing`、
`actor_critic_norm_independence`、`normalizer_checkpoint_restore` 五个实现级来源项，并补充
停止阈值、首次观测/缓存语义。它们都归属引擎语义阻塞项；框架证据已确认，PACE 身份仍未决。

## 当前环境实际送出的张量

对照 [配置](../src/pace_stage1/config.py)、[观测构造](../src/pace_stage1/semantics.py)和
[环境调用](../src/pace_stage1/env.py)，策略侧顺序是物理量拼接、固定尺度、缩放后单位噪声、环境 ±100 裁剪。

| 分量 | 固定尺度 | 缩放后均匀噪声半幅 | 等效缩放前半幅 |
|---|---:|---:|---:|
| 基座线速度 | 2 | 0.2 | 0.1 m/s |
| 基座角速度 | 0.25 | 0.05 | 0.2 rad/s |
| 重力投影 | 1 | 0.05 | 0.05 |
| 命令 | (2,2,0.25) | 0 | 0 |
| 编码器关节位置 | 1 | 0.01 | 0.01 rad |
| 关节速度 | 0.05 | 0.075 | 1.5 rad/s |
| 上一步环境动作 | 1 | 0 | 0 |

这里的数值是当前实现事实，不是 PACE 公开噪声幅值。
价值侧使用单独构造、没有噪声的 48 维部分；拼接力、力矩、摩擦、接触以及 294 维高程。
高程先以米为单位裁剪到 ±1，再乘 5；拼接后整体再裁剪到 ±100。

自建归一化器如以后接入，应明确收到的是上述哪个张量；不能把环境 ±100 裁剪和归一化器输出 ±100 裁剪混为一谈。
目前 `eval()`（评估模式）也不会阻止调用者显式传 `update=True`；其统计恢复组件测试通过，
并不代表真实运行器已经完成保存/恢复接线。

## 尺度抵消的准确边界

对正尺度和一致变换的统计量，零 epsilon、无裁剪且噪声同步变换时：

```math
\frac{sx-s\mu}{s\sigma}=\frac{x-\mu}{\sigma},\qquad s>0,\quad\sigma>0.
```

这个等式不要求“统计收敛”；同一批有限样本的均值和标准差也具有这种尺度协变性。
早期差异来自未同步缩放的初始统计、伪计数、稳定性常数、裁剪或不同输入/更新时机。

对官方对照模块的分母，固定尺度改变有效 epsilon：

```math
\frac{sx-s\mu}{s\sigma+\epsilon}
=\frac{x-\mu}{\sigma+\epsilon/s}.
```

当前自建分母则对应 `epsilon/s²`。因此不能只用尺度抵消论证跳过噪声、裁剪、常数维度、
价值侧高程和首次统计更新审计。

## 后续决议所需内容

下一步应形成一次观测协议预注册，明确选择一个官方版本的归一化语义作为移植依据，
同时列出任何偏离，包括首次观测是否先更新、停止阈值、缓存张量和推理恢复。
这属于向固定 v1.0.2 基础框架移植组件的复现选择，不能宣称已确认论文版本，
也不需要升级整个 PPO（近端策略优化）框架。

固定尺度继续保留为基础框架派生假设，不依据运动表现选择；与当前噪声幅值/顺序及环境裁剪一起登记。
选定这些语义后再修改引擎和运行器，执行组件与恢复一致性验证，才可关闭三个观测阻塞项。

当前仍为 6 个语义阻塞项（观测 3、奖励 3）和 1 个可复现性阻塞项。
短程验证与正式训练门禁均关闭；本次只读追查和来源拆分完成，观测语义实现尚未完成。
