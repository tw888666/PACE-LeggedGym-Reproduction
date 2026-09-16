# 物理容量修复后成对最终评估执行记录

2026-09-13，用户要求“开始评估”，执行两个容量修复后模型的既定最终评估。两个训练均完成 30000 次、原生监督器正常退出且未记录原生错误。

## 模型和评估安排

模型来自 `artifacts/ppo/gpt-pace-v2-formal-stage1-seed0-capacity-r2/gpt-model-30000.pt` 与对应 Stage2 目录中的同名检查点。禁止使用历史模型作本次 Stage2 的直接基线，禁止挑选中间最好检查点。

完全复用 [既定协议](gpt-Stage1-PACEv2-最终评估协议-v1.md) 及样本计划：每模型 28800 个主回合、576 个平地参照、48 段固定视频；共 58752 个定量回合和 96 段视频。确定性策略均值推理，恢复归一化统计后禁止更新。五类地形、十个难度、三个指令层和三个评估种子保持不变。此工作不更新模型权重。

GPU2/3 分别运行 Stage1/2 定量评估，GPU0/1 分别录制 Stage1/2 视频。每个评估分组沿用独立进程及原生日志审核。定量和视频全部完成后，顺序生成 Stage1 报告、Stage2 报告与成对比较，避免 Stage2 引用尚未完成的对照。任一工作进程失败将阻止生成最终完成结论，其他已运行进程可继续保留结果。

本次仅修正报告中的数据来源及历史叙述：旧模板写死的第 300 次中断、旧训练物理警告及旧基线路径不适用于新运行。新报告依据本次入口元数据；训练未记录的速度误差曲线标注缺失，不用奖励代替。评估环境、指标计算及阈值未改变。

## 输出与状态

- Stage1：`artifacts/final-evaluation/gpt-stage1-pace-v2-capacity-r2-v1`。
- Stage2：`artifacts/final-evaluation/gpt-stage2-pace-v2-capacity-r2-v1`。
- 总体进度：`artifacts/final-evaluation/gpt-容量修复成对评估-v1/gpt-评估总体进度.json`。
- tmux（可断线终端）会话：`gpt-capacity-r2-eval`。

单行启动命令：

```bash
tmux new-session -d -s gpt-capacity-r2-eval -c /home/xy.chen/tw/PACE-LeggedGym-Reproduction 'PYTHONPATH=src /home/xy.chen/miniconda3/bin/conda run --no-capture-output -n bruce_gym python -u -m pace_stage1.paired_repaired_evaluation > artifacts/final-evaluation/gpt-容量修复成对评估入口.log 2>&1'
```

启动核验已完成：两个模型分别完成首个 576 回合定量分组及首个 3 段视频分组，四组均通过原生日志审核，归一化统计保持冻结。后续分组继续运行，实时完成数见总体进度文件。

下一步：持续执行全部既定样本，自动生成两阶段报告与成对比较。报告和视频文件核验不等同于人工审阅全部步态；完成后据固定视频做行为解释，不自动启动 Stage3。

## 报告生成恢复（2026-09-13）

两个阶段各 51 个定量分组及 16 个视频分组已全部完成，四个工作进程均退出 0；每阶段 29376 个回合、48 段视频。随后报告进程因 `bruce_gym` 缺少 pandas（数据分析库）而失败，失败发生在读取汇总数据之前，不影响已审核的仿真结果。

恢复使用前次已有分析环境 `/tmp/gpt-pace-final-analysis.TQ0xur/bin/python`，版本为 NumPy 1.23.5、pandas 1.5.3、Matplotlib 3.7.5。未安装或升级仿真环境依赖；仅重新运行报告、训练趋势和视频文件核验。调度入口新增 `--analysis-python` 指定分析解释器与 `--report-only` 仅恢复报告模式，并在后续启动前检查分析依赖。原始失败日志及恢复前状态保留。

本次恢复命令：

```bash
PYTHONPATH=src python -u -m pace_stage1.paired_repaired_evaluation --report-only --analysis-python /tmp/gpt-pace-final-analysis.TQ0xur/bin/python
```

## 完成核验与结论

报告恢复已正常退出，总体状态为 `COMPLETED`。两个阶段的定量样本、精确动作尾部、评估随机性置信区间、训练趋势图及各 48 段固定视频文件核验均完成。Stage2 报告明确引用本次修复后 Stage1 结果，没有混用历史基线。

| 主样本指标 | 修复后 Stage1 | 修复后 Stage2 |
|---|---:|---:|
| 线速度均方根误差 m/s | 0.12578 | 0.11017 |
| 偏航均方根误差 rad/s | 0.09853 | 0.06015 |
| 生存率 | 97.9757% | 98.4340% |
| 联合任务成功率 | 97.5417% | 96.3889% |
| 原始策略均值平均绝对值 | 30.37976 | 0.68977 |
| 力矩饱和率 | 75.6443% | 0.4760% |
| 诊断功率 W | 2275.3942 | 182.9383 |

两阶段均达到既定任务操作性标准。Stage2 的诊断功率下降约 91.96%，动作尾部和饱和显著降低，但联合成功率下降约 1.15 个百分点，主要差异在楼梯与障碍地形。这与能耗奖励兼有抑制激进控制的作用相一致；每阶段仅一个训练种子，不能表述为多种子重复验证。诊断功率不含势能修正及指令速度归一化，不等同于电池功率。

另抽查固定平地 case 28928（0.75 m/s 前进）的 5、10、15 秒画面，截图保存在两个结果目录的 `gpt-行为核查`。这是静态抽查，不足以判定全地形步态自然，也没有把全部 96 段视频宣称为人工审阅完成。

最终报告：[Stage1](../artifacts/final-evaluation/gpt-stage1-pace-v2-capacity-r2-v1/gpt-最终报告/gpt-Stage1-PACEv2-最终科学评估报告.md) · [Stage2 及成对比较](../artifacts/final-evaluation/gpt-stage2-pace-v2-capacity-r2-v1/gpt-最终报告/gpt-Stage2-PACEv2-最终科学评估报告.md)。本次最终评估已完成；下一步审阅固定视频和困难地形结果，再讨论 Stage3 的预算与进入条件，不改回 Stage1/2 参数。
