# Stage1.2 硬裁剪条件下策略正则化实验协议

## 目的

Stage1.2 只处理 `[-1,1]` 动作裁剪之外的策略分布退化。所有实验保持 Stage1 的
PACE actuator（执行器）、PD（比例—微分）参数、电机包络、3 步延迟、任务奖励、
FTD（足端触地）课程和 PPO（近端策略优化）配置不变。Energy reward（能耗奖励）和
torque L2（力矩二范数）均关闭。

训练惩罚定义为：

$$
r_{\mathrm{act}}=-\lambda_a
\frac{1}{12}\sum_{j=1}^{12}a_{\mathrm{raw},j}^2.
$$

这里的 $a_{\mathrm{raw}}$ 是裁剪前的 PPO 采样动作；执行器只接收
$\operatorname{clip}(a_{\mathrm{raw}},-1,1)$。

## 第一轮筛选

五组均使用 seed 0、4096 个环境、平面地形和 500 次迭代：

| 组别 | $\lambda_a$ |
| --- | ---: |
| A0 | 0 |
| A1 | $10^{-5}$ |
| A2 | $3\times10^{-5}$ |
| A3 | $10^{-4}$ |
| A4 | $3\times10^{-4}$ |

训练沿用 Stage1 command distribution（命令分布）、摩擦随机化和 push（推扰）。每组
训练后使用固定 $v_x=1\,\mathrm{m/s}$、无观测噪声、无推扰的 20.01 秒确定性评估。

## 指标边界

训练日志按 iteration 记录：

- actor mean（策略均值）的绝对值、均方根与越界比例；
- sampled raw action（采样原始动作）的绝对值、均方根、最大值与裁剪比例；
- executed action（执行动作）的绝对值和均方根；
- 总体与逐关节裁剪比例；
- 目标角范围和 URDF（统一机器人描述格式）限位违反比例；
- 力矩饱和率和平均力矩利用率；
- 速度与偏航角速度 RMSE（均方根误差）；
- episode length（回合长度）和超时存活率。

固定速度评估中的任务成功定义为：

```text
survived_to_timeout
AND episode_velocity_RMSE <= 0.25 m/s
```

`timeout_survival_rate` 只表示存活，不再命名为 success rate（成功率）。

## 候选冻结门槛

| 类别 | 门槛 |
| --- | ---: |
| 固定速度总体 RMSE | 不超过 0.20 m/s |
| 超时存活率 | 不低于 95% |
| 联合任务成功率 | 不低于 90% |
| 采样动作裁剪比例 | 小于 10%，目标小于 5% |
| 目标角限位违反率 | 0 |
| 力矩饱和率 | 小于 10% |
| policy std | 后期稳定且不持续增长 |

满足门槛还不等于自动冻结；最终候选必须确认没有静止解、暴力甩腿或持续依赖裁剪平台。
