# 物理容量修复后成对重跑准备

状态更新（2026-09-11）：用户已批准在 GPU1 顺序执行成对重训，实际队列与输出见 [执行记录](gpt-物理容量修复成对重训执行记录.md)。下文保留启动前准备方案；其中 GPU2/GPU3 独立启动命令不是本次执行命令。完整工程测试结果见 [容量排查](gpt-Stage2-物理容量排查结果.md)。

测试已收尾：候选组合的 A100 集中压力测试完整 2000 步（九次同步重置）通过，V100 600 步（两次同步重置）复核通过，均无原生警告；初始状态对照、组件测试、奖励接线和原生监督器测试通过。通过的是这些有限检查，不保证未来30k全程绝无容量问题，因此仍需原生监督。

## 范围

若进入下一次正式运行，Stage1 与 Stage2 应使用相同的构造期分离、接触对上限 `2**25`、总缓冲倍率 5。从头 seed0、4096 混合地形、30000 次更新，动作、Eq.(9)、归一化、PPO（近端策略优化）、熵调度、任务奖励和最终评估全部沿用。Stage2 继续使用已确认的负势能解释及原能耗课程。

更改的是仿真构造与内存容量，不是任务调参。初始可观测状态对照相同，但纠正遗漏交互后，后续物理轨迹和模型权重可能变化。因此不能从旧模型接着训练来清除历史风险，也不覆盖已有实验目录。

## 已准备的单行命令

以下不是已执行记录。采用独立目录与 tmux（可断线终端）会话，且由原生错误监督器启动训练；再出现遗漏交互即停止该运行。GPU2、GPU3 可按届时安排选择，不改全局环境变量。

Stage1：

```bash
tmux new-session -d -s gpt-stage1-capacity-r2 -c /home/xy.chen/tw/PACE-LeggedGym-Reproduction 'CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src conda run --no-capture-output -n bruce_gym python -m pace_stage1.native_training_supervisor --stage 1 --output artifacts/ppo/gpt-pace-v2-formal-stage1-seed0-capacity-r2 > artifacts/ppo/gpt-stage1-capacity-r2-监督入口.log 2>&1'
```

Stage2：

```bash
tmux new-session -d -s gpt-stage2-capacity-r2 -c /home/xy.chen/tw/PACE-LeggedGym-Reproduction 'CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src conda run --no-capture-output -n bruce_gym python -m pace_stage1.native_training_supervisor --stage 2 --output artifacts/ppo/gpt-pace-v2-formal-stage2-seed0-capacity-r2 > artifacts/ppo/gpt-stage2-capacity-r2-监督入口.log 2>&1'
```

这里 r2（第二次工程执行）区分原始运行，不表示按步态或回报挑选更佳参数。两个阶段均为新的独立从头运行；是否并行或先后执行不改变各自目标和参数。正式启动仍需明确进入这一步，当前请求下只完成定位、代码修复准备与无训练测试。

旧模型的最终评估和步态改善结论保留为描述性证据；新运行完成后复用同一最终模型评估样本，不以中间最佳检查点替代 30000 次模型。Stage3 仍未开放。
