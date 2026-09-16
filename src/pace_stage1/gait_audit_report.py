"""Read-only plots and Chinese descriptive report for fixed gait replays."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def run(root,stage=1,repaired_run=False,baseline_root=None):
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    names=dict(smooth_slope='光滑坡',rough_slope='粗糙坡',stairs_negative_step='负台阶楼梯',stairs_positive_step='正台阶楼梯',boxes='离散障碍',flat_reference='平地参照')
    rows=[]
    for path in sorted(root.glob('gpt-case-*/gpt-步态结果.json')):
        data=json.loads(path.read_text())
        if not data.get('runtime_log_checked'):raise RuntimeError('unaudited native log')
        rows.append(data)
    if len(rows)!=6:raise RuntimeError('expected six fixed cases')
    rows.sort(key=lambda x:int(x['case']['case_id']))
    fig,axes=plt.subplots(6,2,figsize=(14,17))
    lines=['# Stage1 固定样本步态描述性分析','',
           '本次为事后描述性回放，不修改 Stage1 最终结果或通过阈值。固定选取五类地形难度 5 和平地参照的正常移动、种子 200000、重复 0，共六个回合。没有执行训练。','',
           '| 地形／回合 | 关节速度均方根 rad/s | 饱和比例 | 支撑比例范围 | 接触事件频率范围 Hz | 接触水平足速代理均值范围 m/s |',
           '|---|---:|---:|---:|---:|---:|']
    def span(values):
        values=[x for x in values if x is not None]
        return '缺失' if not values else '%.2f–%.2f'%(min(values),max(values))
    for i,row in enumerate(rows):
        c=row['case'];m=row['metrics'];legs=m['legs'];label=names[c['terrain']]
        lines.append('| %s／%s | %.2f | %.1f%% | %s | %s | %s |'%(label,c['case_id'],m['joint_velocity_rms_all_rad_s'],100*m['torque_saturation'],span([x['duty_factor'] for x in legs]),span([x['touchdown_event_hz'] for x in legs]),span([x['contact_horizontal_speed_mean_m_s'] for x in legs])))
        arrays=np.load(root/('gpt-case-%s'%c['case_id'])/'gpt-步态时序.npz')
        t=(np.arange(len(arrays['contact']))+1)*.01
        mask=(t>=5)&(t<=8)
        if not mask.any():mask=np.ones_like(t,dtype=bool)
        legnames=row.get('foot_body_names') or ['足1','足2','足3','足4']
        axes[i,0].imshow(arrays['contact'][mask].T,aspect='auto',interpolation='nearest',cmap='Greys',vmin=0,vmax=1,extent=[t[mask][0],t[mask][-1],3.5,-.5])
        axes[i,0].set_yticks(range(4));axes[i,0].set_yticklabels(legnames)
        axes[i,0].set_title(label+'：小腿合并刚体接触序列（黑=接触）')
        axes[i,1].plot(t[mask],arrays['qdot'][mask],linewidth=.7,alpha=.7)
        axes[i,1].set_title(label+'：12 关节速度');axes[i,1].set_ylabel('rad/s')
        for ax in axes[i]:ax.set_xlabel('回放时间（秒）')
    fig.tight_layout();fig.savefig(root/'gpt-固定步态接触与关节速度.png',dpi=150);plt.close(fig)
    lines+=['','## 各腿接触比例','',
            '| 地形 | 左前 LF | 左后 LH | 右前 RF | 右后 RH |',
            '|---|---:|---:|---:|---:|']
    for row in rows:
        by_name=dict(zip(row['foot_body_names'],row['metrics']['legs']))
        lines.append('| '+names[row['case']['terrain']]+' | '+' | '.join('%.2f%%'%(100*by_name[n+'_SHANK']['duty_factor']) for n in ('LF','LH','RF','RH'))+' |')
    flat=next(row for row in rows if row['case']['terrain']=='flat_reference')
    flat_arrays=np.load(root/('gpt-case-%s'%flat['case']['case_id'])/'gpt-步态时序.npz')
    heights=np.quantile(flat_arrays['foot_position'][100:,:,2],[.05,.5,.95],axis=0)
    lines+=['','## 平地几何位置交叉检查','',
            '同一固定平地回放，从第 1 秒起统计足底球形碰撞中心的世界高度；地面为 z=0，足底球半径约 0.0315 m。这是排除接触读数单一证据的描述性补充。','',
            '| 腿 | 高度 p05 (m) | 中位数 (m) | p95 (m) |','|---|---:|---:|---:|']
    for j,name in enumerate(flat['foot_body_names']):
        lines.append('| %s | %.3f | %.3f | %.3f |'%(name,*heights[:,j]))
    lines+=['',
            '本次固定平地样本的左后 LH 与右前 RF 接触比例为零，足端球心高度显著高于地面；左前 LF 与右后 RH 则周期性接近地面。这支持该回放主要依靠一对对角腿支撑、另外两腿抬起的解释。它比泛称“高频甩腿”更具体，但不能据六个回合推断所有 28800 回合都采用同一步态。',
            '这项观察保留为 Stage1 的 B 类控制行为结果，不追加对称性奖励、动作惩罚或新的通过门槛。Stage2 仍只加入能耗奖励，检验这些控制及接触模式是否随之变化。',
            '','## 解释限制','',
            '接触信号来自合并后的整个小腿刚体，不等价于独立足底传感器。原始接触开始事件可能包含抖动，不能把事件频率直接称为稳定步频；每条腿的完整周期、支撑及摆动段时长见各回合 JSON（结构化数据）。',
            '水平足速范围可能包含仅一次接触的抬起腿，必须结合各腿接触比例和事件数阅读；不能据这种稀疏样本推断全程严重滑脚。',
            '水平足速来自足底碰撞球心位置的差分，只是滑脚代理。坡面局部切向速度、滚动接触点速度和真正滑移速度并不相同。没有温和步态对照，不能只凭这些绝对数值断言高于正常 ANYmal 步频。',
            '回放固定采用相同模型、样本随机流及观测规范，但单实例运行可能与最终批量评估有浮点轨迹差异；本表只用于机制解释，不替换 28800 回合科学评估。',
            '图固定展示 5–8 秒，没有按异常程度选择时间窗。后续 Stage2 使用完全相同的程序与样本比较，而非改变 Stage1 奖励。','',
            '[接触与关节速度图](gpt-固定步态接触与关节速度.png)','']
    if stage==2:
        lines[0]='# Stage2 固定样本步态描述性分析'
        lines[2]='沿用 Stage1 的六个固定回放样本与诊断定义，不训练、不挑选步态或更换时间窗。Stage2 训练日志存在可能遗漏交互的原生警告，以下只描述该模型行为，不作为干净的复现结论。'
        # A Stage1 behavioral finding must never be asserted for a new model.
        lines=[line for line in lines if not line.startswith('本次固定平地样本的左后') and not line.startswith('这项观察保留为 Stage1')]
        if repaired_run and baseline_root is None:
            raise ValueError('repaired comparison requires an explicit repaired baseline')
        baseline=baseline_root or Path(__file__).resolve().parents[2]/'artifacts/gait-audit/gpt-stage1-model30000-v1'
        lines+=['','## 与相同 Stage1 固定回合比较','',
                '| 地形 | Stage1 关节速度 RMS（均方根）rad/s | Stage2 RMS rad/s | Stage1 饱和 | Stage2 饱和 |',
                '|---|---:|---:|---:|---:|']
        for row in rows:
            c=row['case'];m=row['metrics']
            old_data=json.loads((baseline/('gpt-case-%s'%c['case_id'])/'gpt-步态结果.json').read_text())
            if repaired_run and ('capacity-r2' not in old_data['checkpoint'] or 'capacity-r2' not in row['checkpoint']):
                raise ValueError('repaired comparison cannot use historical checkpoint')
            old=old_data['metrics']
            lines.append('| %s | %.3f | %.3f | %.2f%% | %.2f%% |'%(names[c['terrain']],old['joint_velocity_rms_all_rad_s'],m['joint_velocity_rms_all_rad_s'],100*old['torque_saturation'],100*m['torque_saturation']))
        lines+=['','支撑是否恢复为四腿、是否仍有抬腿，需要结合各腿接触比例、平地球心高度及连续时间序列判断；不预设 Stage2 已改善。六个回合不能代表总体任务成功率。',
                '训练物理有效性问题独立于这里的评估原生日志核验。详见项目 provenance/gpt-Stage2-训练完成与原生警告审计.md。','']
    if repaired_run:
        lines[0]='# Stage%d 容量修复后固定样本步态描述性分析'%stage
        lines[2]='沿用既定六个固定回放样本和 5–8 秒窗口，使用本次容量修复后 30000 次模型，不训练、不调整参数。训练与回放日志均未检测到原生物理错误；单实例回放仍只作描述性解释，不替换批量最终评估。'
        excluded=('本次固定平地样本的左后','这项观察保留为 Stage1','训练物理有效性问题独立于')
        lines=[line for line in lines if not line.startswith(excluded)]
        lines=['各腿支撑模式及抬腿情况需要结合接触比例、平地球心高度和连续时序判断；六个回放不能代表总体任务成功率。' if line.startswith('支撑是否恢复为四腿') else line for line in lines]
        lines+=['','本次实际检查点：`%s`。'%rows[0]['checkpoint'],
                '不继承历史模型的两腿支撑结论；以本次接触比例、足端高度及固定视频为准。']
        if stage==2:lines+=['本次 Stage1 对照目录：`%s`。'%baseline]
    (root/('gpt-Stage%d-步态描述性分析报告.md'%stage)).write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',type=int,choices=(1,2),default=1);p.add_argument('--repaired-run',action='store_true');p.add_argument('--baseline-root',type=Path);a=p.parse_args();run(a.root,a.stage,a.repaired_run,a.baseline_root)
