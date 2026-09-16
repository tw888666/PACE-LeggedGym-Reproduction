"""Summarize engineering capacity diagnostics without interpreting task returns."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
CASES=[
 ('原始布局，倍率5，接触上限2**23','gpt-stage2-capacity-v1/gpt-natural-buffer-5'),
 ('原始布局，倍率20，接触上限2**23','gpt-stage2-capacity-natural20/gpt-natural-buffer-20'),
 ('集中布局，倍率5，接触上限2**23','gpt-stage2-capacity-v1/gpt-concentrated-buffer-5'),
 ('集中布局，倍率20，接触上限2**23','gpt-stage2-capacity-v1/gpt-concentrated-buffer-20'),
 ('构造分离，原始课程，上限2**23','gpt-stage2-capacity-staged/gpt-staged-buffer-5'),
 ('构造分离，集中压力，上限2**23','gpt-stage2-capacity-staged-stress/gpt-staged_concentrated-buffer-5'),
 ('原始布局，倍率5，上限2**25','gpt-stage2-capacity-pairs32m/gpt-natural-buffer-5'),
 ('构造分离，集中压力，上限2**25','gpt-stage2-capacity-staged-stress-pairs32m/gpt-staged_concentrated-buffer-5'),
 ('V100：构造分离，集中压力，上限2**25','gpt-stage2-capacity-v100-confirmation/gpt-staged_concentrated-buffer-5')]


def report():
    base=ROOT/'artifacts/physics-audit';rows=[];completed=0
    for label,relative in CASES:
        directory=base/relative;marker=directory/'gpt-审核结果.json'
        if not marker.exists():rows.append('| %s | 运行中 | 待定 | — |'%label);continue
        d=json.loads(marker.read_text());completed+=1
        finished=(directory/'gpt-子进程结果.json').exists() and d['returncode']==0
        steps=str(d['steps']) if finished else '提前停止，未完成计划步数'
        rows.append('| %s | %s | %s | %d |'%(label,'无原生警告' if d['runtime_clean'] else '出现原生警告／执行失败',steps,len(d['native_warnings'])))
    text='''# Stage2 物理容量排查结果

## 结论与证据

当前配置的原生聚合候选容量溢出已在无 PPO（近端策略优化）更新的测试中复现。细化阶段日志显示，第一次 15713824 候选容量请求发生在 prepare_sim（准备仿真）阶段，早于初始重置及策略推理；原始布局首次策略步又触发第二条警告。数值与历史训练日志相符，说明无需长训即可触发；不是仅凭日志末尾位置推断的析构错误。

4096 个机器人创建时共享 120 个起点，单点最多 47 个，按起点计算有 69383 对机器人组合。集中压力测试为 20 个起点、单点最多 205 个。这个计数不是 PhysX 内部候选条数，但与分离构造位置的干预结果共同支持初始化重合是触发因素。

单独把 default_buffer_size_multiplier（总缓冲倍率）从 5 提到 20 无效。单独分离构造位置能消除原始布局短程警告，但集中压力仍触发后续警告。候选修复组合为：构造期分离、首次重置前恢复原起点、max_gpu_contact_pairs（接触对上限）从 2**23 增至 2**25，总倍率仍为 5。

## 测试状态

| 测试 | 结果 | 完成策略步 | 捕获警告行数 |
|---|---|---:|---:|
%s

已完成 %d/%d 项。警告行数不是错误影响的回合数，不能推算丢失接触数量。所有测试只回放冻结模型，不更新权重。出现已复现原生错误的提前停止测试不计为完整 2000 步通过。V100 的 600 步复核覆盖两次同步重置；2000 步集中测试覆盖九次。

## 初始状态与奖励检查

构造分离前后，4096 个环境的机身状态、关节状态、48维策略观测、353维价值观测、原始起点、环境随机生成器状态和 CPU PyTorch 随机状态逐张量相同。单独接触对扩容也通过同一初始状态对照。未声称 GPU 内部物理状态或整个后续随机轨迹逐位相等。

候选组合模块 `construction_staging.py` 的三项组件检查通过；V100 上八环境固定动作的 Stage2 奖励测试通过，覆盖课程起点、半衰期和终止回合记账，没有 PPO 更新。

## 对已有结果的影响

Stage1 保存的原始及第300次续训源码，与此次复现采用的 _create_sim、_create_envs、_make_origins 方法相同。初始化风险可能跨两个阶段；没有完整原生时间证据时，不能把 Stage1 当作已经排除该风险的无异常对照，也不能推定历史每个回合全部失效。

现有 Stage1/Stage2 模型及最终评估全部保留，模型行为比较仍可描述，但不足以作为排除仿真问题后的能耗奖励因果结论。任何新测试成功都不能追溯修复旧训练轨迹。

## 可执行修复与下一步

两个正式启动器新增显式 `--construction-capacity-repair` 选项，默认仍是旧路径。候选组合仅改变构造期临时位置及缓冲容量，不改变奖励、PPO、动力学参数、地形或初始观测；然而消除遗漏交互可能改变后续轨迹和最终权重，因此新训练必须放入独立目录，从头开始，不接续旧模型。

另备 `native_training_supervisor`（原生训练日志监督入口），通过 stdbuf 行缓冲及时捕获原生输出。再出现遗漏交互、无效物理参数或 CUDA 错误即停止该子进程，保留失败状态；三项监督器测试通过，包括原生 C printf 输出警告后休眠的子进程，在退出前即被捕获并停止。

在压力检查完成后，下一步建议分别对 Stage1 与 Stage2 做同一工程修复下的 seed0 从头正式重跑，保留两阶段唯一学习目标差异为能耗项。当前尚未启动任何重训或 Stage3。有限压力测试通过也不保证整个30k永不溢出，原生监督必须随新运行启用。

方案与适应性补充见 [诊断方案](%s)。
'''%('\n'.join(rows),completed,len(CASES),ROOT/'provenance/gpt-Stage2-物理容量诊断方案.md')
    (ROOT/'provenance/gpt-Stage2-物理容量排查结果.md').write_text(text)
    (base/'gpt-物理容量排查状态.json').write_text(json.dumps(dict(completed_tests=completed,required_tests=len(CASES),state='COMPLETED' if completed==len(CASES) else 'RUNNING'),indent=2)+'\n')
    return completed==len(CASES)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    while not report():
        if not a.wait:break
        time.sleep(30)
