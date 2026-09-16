"""Validate fixed video coverage and write the final scientific handoff."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def repaired_report(root,stage,entry,summary,table):
    """Describe this paired rerun using its explicit baseline, never the old model."""
    m=summary['metrics']
    comparison=[]
    if stage==2:
        baseline_root=Path(entry['baseline_root'])
        baseline_entry=json.loads((baseline_root/'gpt-评估入口.json').read_text())
        if not baseline_entry.get('construction_capacity_repaired'):
            raise RuntimeError('paired repaired report requires the repaired Stage1 baseline')
        baseline=json.loads((baseline_root/'gpt-最终报告/gpt-最终汇总.json').read_text())
        comparison=['| 指标 | 修复后 Stage1 | 修复后 Stage2 |','|---|---:|---:|']
        for key,title in [('velocity_rmse','线速度均方根误差 m/s'),('yaw_rmse','偏航均方根误差 rad/s'),('survival','生存比例'),('joint_task_success','联合任务成功比例'),('mean_abs_mu','原始策略输出平均绝对值'),('prob_abs_mu_gt_5','回合等权原始输出绝对值 >5 比例'),('saturation','力矩饱和比例'),('diagnostic_power_w','同口径诊断功率 W')]:
            comparison.append('| %s | %.5f | %.5f |'%(title,baseline['metrics'][key],m[key]))
        comparison.extend(['','对照来源：`%s`。'%baseline_root])
    return '''# Stage%d 物理容量修复后最终科学评估

本次评估对应 `%s`，使用从头 seed0 完成的第 30000 次更新模型。两个阶段均采用构造期分离及相同接触对容量修复，在 GPU1 顺序完成训练；原生监督器正常退出，未检测到遗漏交互等原生错误。这是已检查日志范围内的结论，不代表证明仿真不存在任何误差。旧运行的物理警告和 Stage1 第 300 次中断不属于本次运行历史。

主样本 28800 回合，另有 576 个平地参照。48 段固定视频已通过来源、帧数及帧率核验；文件核验不等同于人工审阅全部步态。

## 任务性能与控制

五类地形、十个难度、三个指令层等权平均：线速度 RMSE（均方根误差）%.3f m/s，偏航 RMSE %.3f rad/s，生存率 %.2f%%，联合成功率 %.2f%%。任务操作性阈值：%s。

联合成功要求完成时限且线速度和偏航 RMSE 各不超过 0.30；主总体标准为联合成功率至少 80%%、生存率至少 90%%。这是项目预注册阈值，不是 PACE 作者阈值。不以动作大小或饱和率设额外通过门槛。

%s

%s

## 解释边界与产物

两个阶段各只有一个训练随机种子；三个评估种子的置信区间只反映评估随机性，不代表重复训练不确定性。最终评估采用确定性策略均值，不能与训练采样动作混称；失败回合误差必须结合生存率和时长解读。

诊断功率为 0.0192×施加力矩平方和，加净机械功率正部；不含势能项及指令速度归一化，不等同于 Stage2 完整训练能耗奖励，也不等同于电池功率。Stage2 负势能解释仍属于用户批准的复现假设。

各地形、难度、指令、种子、逐关节和精确动作尾部见本目录 CSV（逗号分隔数据文件）。[固定视频索引](gpt-固定视频索引.md) · [地形分层图](gpt-最终地形分层.png)。训练趋势另存于上一级 `gpt-训练趋势`。

保留所有固定样本和失败片段，不按结果重调参数。下一步结合分层结果和固定视频解释能耗与控制行为差异；本报告不自动启动 Stage3。
'''%(stage,entry['checkpoint'],m['velocity_rmse'],m['yaw_rmse'],100*m['survival'],100*m['joint_task_success'],'达到' if summary['task_adequate'] else '未达到','\n'.join(table),'\n'.join(comparison))


def run(root,stage=1):
    import pandas as pd
    output=root/'gpt-最终报告'
    summary=json.loads((output/'gpt-最终汇总.json').read_text())
    entry_path=root/'gpt-评估入口.json'
    entry=json.loads(entry_path.read_text()) if entry_path.exists() else {}
    archive_path=root/'gpt-视频归档清单.json'
    archive=json.loads(archive_path.read_text()) if archive_path.exists() else {}
    archived_videos={r['original_path']:r['archive_path'] for r in archive.get('archived',[])}
    with (ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv').open() as stream:
        cases=list(csv.DictReader(stream))
    planned={c['case_id']:c for c in cases if c['seed']=='200000' and c['replicate']=='0'
             and (c['difficulty_row'] in ('2','5','8') or c['terrain']=='flat_reference')}
    if len(planned)!=48:
        raise RuntimeError('fixed video case count mismatch')
    records=[]
    for marker in sorted((root/'videos').glob('*/gpt-完成记录.json')):
        metadata=json.loads(marker.read_text())
        if not metadata.get('runtime_log_checked') or not metadata.get('normalization_frozen'):
            raise RuntimeError('video group has not passed runtime verification')
        rows=json.loads((marker.parent/'gpt-逐回合指标.json').read_text())
        for row in rows:
            if row['case_id'] not in planned or any(str(row[k])!=v for k,v in planned[row['case_id']].items()):
                raise RuntimeError('video metadata differs from fixed case')
            path=marker.parent/('gpt-case-%s.mp4'%row['case_id'])
            if not path.exists() and str(path.resolve()) in archived_videos:
                path=Path(archived_videos[str(path.resolve())])
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'stream=nb_frames,width,height,r_frame_rate','-show_entries','format=duration','-of','json',str(path)],text=True))
            stream=probe['streams'][0]
            if stream['r_frame_rate']!='25/1' or int(stream['nb_frames'])!=(row['steps']+3)//4:
                raise RuntimeError('video frame count or rate differs from recorded episode')
            records.append(dict(case_id=row['case_id'],terrain=row['terrain'],difficulty_row=row['difficulty_row'],
                command_bin=row['command_bin'],duration_s=float(probe['format']['duration']),
                path=str(path.resolve()),steps=row['steps']))
    if len(records)!=48 or {r['case_id'] for r in records}!=set(planned):
        raise RuntimeError('not all 48 preregistered videos are complete')
    pd.DataFrame(records).to_csv(output/'gpt-固定视频核验.csv',index=False)
    names={'smooth_slope':'光滑坡','rough_slope':'粗糙坡','stairs_negative_step':'负台阶楼梯',
           'stairs_positive_step':'正台阶楼梯','boxes':'离散障碍','flat_reference':'平地参照'}
    terrain=pd.read_csv(output/'gpt-地形统计.csv')
    table=['| 地形 | 线速度 RMSE (m/s) | 偏航 RMSE (rad/s) | 生存率 | 联合成功率 | 饱和比例 | 诊断功率和 (W) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in terrain.iterrows():
        table.append('| %s | %.3f | %.3f | %.2f%% | %.2f%% | %.2f%% | %.1f |'%(names[r.terrain],r.velocity_rmse,r.yaw_rmse,100*r.survival,100*r.joint_task_success,100*r.saturation,r.diagnostic_power_w))
    m=summary['metrics']
    label='任务达到预注册操作性标准；控制行为 A/B 标签需结合连续指标和固定视频描述。' if summary['task_adequate'] else 'C：当前能耗关闭 Task-only 消融未达到预注册的任务操作性标准。'
    highest_video_row=max(int(r['difficulty_row']) for r in records
                          if r['command_bin']=='normal_motion' and r['difficulty_row'])
    visible_videos=[r for r in records if r['command_bin']=='normal_motion'
                    and r['difficulty_row']==str(highest_video_row)]
    gallery=['# 固定样本视频索引','',
             '仅展示既有固定录像中最高难度行 %d 的 %d 段 normal_motion（正常移动）样本，指令为前进0.75 m/s。'%(highest_video_row,len(visible_videos)),
             '地形配置实际最高为9；本索引原录像覆盖2、5、8，故保留其中最高的8。保留首次终止和失败片段；点击链接可播放。',
             '其余录像不在本索引展示；已归档文件可按评估根目录的 gpt-视频归档清单.json 查找，全部定量评估保持不变。','']
    for r in visible_videos:
        gallery.append('- [%s / 难度 %s / 正常移动 / case %s](%s)，%.2f 秒。'%(names[r['terrain']],r['difficulty_row'] or '无',r['case_id'],r['path'],r['duration_s']))
    (output/'gpt-固定视频索引.md').write_text('\n'.join(gallery)+'\n')
    text='''# Stage1 PACE v2 最终科学评估

本报告只评估全局第 30000 次更新检查点，没有选择中间最佳检查点。主样本 28800 回合，另有 576 个平地参照；48 段固定样本视频全部通过帧率、帧数及来源核验。

## 主结论

%s

五类地形、十个难度和三个指令层等权平均：线速度 RMSE（均方根误差）%.3f m/s，偏航 RMSE %.3f rad/s，生存率 %.2f%%，联合成功率 %.2f%%。
联合成功要求完成时限且线速度/偏航 RMSE 均不超过 0.30；总体操作性标准为联合成功率至少 80%%、生存率至少 90%%。这些是项目预注册阈值，不是 PACE 作者阈值。

%s

## 解释边界

动作分位数和超阈值比例见同目录精确尾部 CSV（逗号分隔数据文件）：最终评估统计的是确定性策略均值；训练趋势统计的是高斯采样动作，不能混称。
逐地形、难度、指令层、种子的结果以及逐关节饱和均已分别输出。失败回合的 RMSE 是终止前误差，必须结合生存率和时长阅读，不能只挑低 RMSE。
能耗是当前冻结的诊断功率：0.0192×施加力矩平方和、净机械功率正部及其诊断功率和；不包含全部势能修正，未用于该 Stage1 奖励。

该训练只有一个 seed（训练随机种子）。置信区间仅反映三个评估种子的回合随机性，不是训练重复的不确定性。
正式训练在第 300 次更新后因检查点校验器错误中断并续训。权重、优化器、归一化统计和更新索引恢复，但模拟器和随机轨迹没有逐位连续恢复；这项边界不能省略。

首批评估曾因实例重合触发 PhysX 遗漏交互警告，已作为无效工程产物保留。此报告仅使用分离起点后的 r2 执行批次，全部分层的原生日志已检查；样本、局部地形、随机输入及阈值未因表现更改。

## 产物与下一步

- 地形、难度、指令和逐关节表：本目录 `gpt-*.csv`。
- 固定视频：[全部 48 段索引](gpt-固定视频索引.md)。
- 最终地形图：[分层科学评估](gpt-最终地形分层.png)。
- 训练图：上一级 `gpt-训练趋势`，与最终评估分开。

保留本次 Stage1 结果，不返回调动作比例、动作惩罚或熵参数。下一步形成 Stage2 能耗项实验的独立执行决议；若为 C，须明确其目标是检验能耗项是否能补足当前 Task-only 消融，不能宣称已有稳定基线。本报告不自动启动 Stage2 或 Stage3。
'''%(label,m['velocity_rmse'],m['yaw_rmse'],100*m['survival'],100*m['joint_task_success'],'\n'.join(table))
    if stage==2 and not entry.get('construction_capacity_repaired'):
        baseline=json.loads((ROOT/'artifacts/final-evaluation/gpt-stage1-pace-v2-v1-r2/gpt-最终报告/gpt-最终汇总.json').read_text())
        comparison=['| 指标 | Stage1 | Stage2 |','|---|---:|---:|']
        for key,title in [('velocity_rmse','线速度均方根误差 m/s'),('yaw_rmse','偏航均方根误差 rad/s'),('survival','生存比例'),('joint_task_success','联合任务成功比例'),('mean_abs_mu','原始策略输出平均绝对值'),('prob_abs_mu_gt_5','回合等权原始输出绝对值 >5 比例'),('saturation','力矩饱和比例'),('diagnostic_power_w','同口径诊断功率 W')]:
            comparison.append('| %s | %.5f | %.5f |'%(title,baseline['metrics'][key],m[key]))
        text='''# Stage2 最终模型评估及与 Stage1 的比较

**训练程序完成，但训练物理有效性存疑。** 完整训练原生日志出现可能遗漏交互的容量警告，其发生更新无法定位。本报告评估该运行产生的原始最终模型，不将其称为干净的 Stage2 科学复现成功；评估自身正常也不能消除训练有效性问题。详见 [训练警告审计](%s)。

采用既定第 30000 次更新检查点，没有选择中间最佳模型。主样本 28800 回合，平地参照 576 回合；48 段固定视频均通过文件、来源、帧率和帧数核验。核验不等同于已人工审阅全部步态。

## 同口径模型表现

%s

数值任务操作性阈值判定：%s。该数值判定与训练仿真有效性分开，不据此开放 Stage3。
联合成功要求完成时限且线速度和偏航均方根误差各不超过 0.30；主任务操作性标准为联合成功率至少 80%%、生存率至少 90%%。

%s

## 解释边界

两个阶段各只有一个训练种子；三个评估种子的置信区间仅反映评估随机性，不代表重复训练不确定性。最终评估采用确定性策略均值；训练采样动作尾部不可直接混用。失败回合误差应结合存活时长阅读。

诊断功率延续 Stage1 口径：0.0192×施加力矩平方和，加净机械功率正部。不含势能项和指令速度归一化，不等同于 Stage2 训练中的完整能耗奖励。Stage2 势能符号采用用户确认的正文解释，属于复现假设。

Stage1 有第 300 次更新中断续训、未逐位恢复模拟器随机轨迹的边界；Stage2 此次无该中断，但存在上述原生物理警告。禁止把这些已知差异隐去后作纯粹的能耗项因果归因。

[固定视频索引](gpt-固定视频索引.md) · [地形分层图](gpt-最终地形分层.png)

保留原模型、日志及全部评估。下一步处理训练碰撞候选容量问题的工程诊断与后续实验决议，不因本报告自动修改协议、重训或启动 Stage3。
'''%(ROOT/'provenance/gpt-Stage2-训练完成与原生警告审计.md','\n'.join(comparison),'达到' if summary['task_adequate'] else '未达到','\n'.join(table))
        summary.update(training_validity='CONCERN_PHYSX_MISSED_INTERACTIONS',
                       task_classification='NUMERICAL_TASK_PASS_WITH_TRAINING_VALIDITY_CONCERN' if summary['task_adequate'] else 'NUMERICAL_TASK_FAIL_WITH_TRAINING_VALIDITY_CONCERN')
        (output/'gpt-最终汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if entry.get('construction_capacity_repaired'):
        text=repaired_report(root,stage,entry,summary,table)
    (output/('gpt-Stage%d-PACEv2-最终科学评估报告.md'%stage)).write_text(text)
    print(json.dumps(dict(status='completed',cases=summary['case_count'],videos=len(records),task_classification=summary['task_classification']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',type=int,choices=(1,2),default=1)
    args=parser.parse_args();run(args.root,args.stage)
