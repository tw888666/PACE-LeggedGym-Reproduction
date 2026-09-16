# 三个阶段平地 1 m/s 步态补充录像

六组均使用第 30000 次更新的最终模型。固定前进指令 1 m/s，横移与偏航指令为零；随机种子 200000。
复用原平地样本 28928 的独立随机流（初态、摩擦与观测噪声），改为单环境回放；仅改变前进指令和标注。编号保留用于随机流复现，本次为补充样本，不并入原 0.75 m/s 正式评估结果。
确定性策略回放，观测归一化统计固定，不更新模型权重。GPU（图形处理器）0、1 录制；每段最长模拟 20.01 秒，25 帧/秒。

[1 m/s 同步对比视频](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-三个阶段平地1米每秒步态同步对比.mp4)

上排：第一阶段、第二阶段、第三阶段 100 W；下排：第三阶段 125、150、175 W。
原速播放；拼图统一裁取原画面中央 320×240 区域并放大两倍。提前终止后显示黑屏与提示，单段录像保留到终止。预算标签是训练阈值，不是片段实测功率。

| 模型 | 单段视频 | 模拟时长（秒） | 存活 | 平移速度均方根误差（m/s） |
|---|---|---:|---|---:|
| 第1阶段 | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-Stage1/gpt-case-28928.mp4) | 20.01 | 是 | 0.1327 |
| 第2阶段 | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-Stage2/gpt-case-28928.mp4) | 20.01 | 是 | 0.1360 |
| 第3阶段 预算100W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-B100-seed0/gpt-case-28928.mp4) | 20.01 | 是 | 0.8935 |
| 第3阶段 预算125W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-B125-seed0/gpt-case-28928.mp4) | 20.01 | 是 | 0.2498 |
| 第3阶段 预算150W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-B150-seed0/gpt-case-28928.mp4) | 20.01 | 是 | 0.9449 |
| 第3阶段 预算175W | [观看](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/artifacts/stage3-budget-search/gpt-local-linear-v1/videos-1mps/gpt-B175-seed0/gpt-case-28928.mp4) | 20.01 | 是 | 0.4001 |

[此前 0.75 m/s 视频入口](/home/xy.chen/tw/PACE-LeggedGym-Reproduction/provenance/gpt-Stage3-平地步态视频入口.md)

这是固定平地样本，不能代替跨地形统计。
