# 三个阶段平地步态视频

补充：[固定前进指令 1 m/s 的六组同步对比与单段视频](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/provenance/gpt-Stage3-平地1米每秒步态视频入口.md)。

所有模型均为第 30000 次更新的最终权重；第三阶段展示已完成的 100、125、150、175 W 四组。
200 W 尚未纳入本次录像。第一、二阶段复用已有回放，第三阶段在 GPU（图形处理器）0、1 录制。

共同设置：平地、前进指令 0.75 m/s、横移与偏航指令为零、随机种子 200000、回合编号 28928。
确定性策略回放，观测归一化统计固定；录制不更新模型权重。回合上限 20.01 秒，25 帧/秒，完整回合为 501 帧，编码后时长 20.04 秒。

[同步对比视频](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-三个阶段平地步态同步对比.mp4)

拼图上排：第一阶段、第二阶段、第三阶段 100 W；下排：第三阶段 125、150、175 W。
同步视频保持原速，统一裁取画面中央 320×240 区域并放大两倍以便观察腿部；单段链接保留原始完整画面。提前终止的回合会在剩余时间显示黑屏和“回合已终止”，不补静止帧。
预算标签表示训练约束阈值，并非该平地片段的实测功率。

| 模型 | 正常前进 | 低速 | 站立 | 前进时长（秒） | 平移速度均方根误差（m/s） |
|---|---|---|---|---:|---:|
| 第1阶段 | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage1-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage1-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage1-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.1331 |
| 第2阶段 | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage2-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage2-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/video-archive/gpt-最高难度筛选-20260916/gpt-stage2-pace-v2-capacity-r2-v1/videos/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.0984 |
| 第3阶段 预算100W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B100-seed0/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B100-seed0/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B100-seed0/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.6926 |
| 第3阶段 预算125W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B125-seed0/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B125-seed0/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B125-seed0/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.4043 |
| 第3阶段 预算150W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B150-seed0/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B150-seed0/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B150-seed0/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.6999 |
| 第3阶段 预算175W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B175-seed0/gpt-50-flat_reference-/gpt-case-28928.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B175-seed0/gpt-50-flat_reference-/gpt-case-28864.mp4) | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos/gpt-B175-seed0/gpt-50-flat_reference-/gpt-case-28800.mp4) | 20.01 | 0.5195 |

提前终止记录：
- 第3阶段 预算175W，低速：15.79 秒终止，原片保留。

此处是单个固定平地样本，不能代替跨地形独立验证结果；步态类型与对称性仍需结合足端接触时序判断。
