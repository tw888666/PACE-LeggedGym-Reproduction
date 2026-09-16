"""Descriptive preregistered training windows, never used for checkpoint selection."""
import json,argparse
from pathlib import Path
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def run(path,output,stage=1,repaired_run=False):
    rows=[]
    with path.open() as stream:
        for line in stream:
            r=json.loads(line);rows.append(dict(iteration=r['iteration'],policy_std=r['std_mean'],tail_gt5=r['raw_action_tail']['prob_abs_gt_5'],saturation=r['torque_saturation_ratio'],velocity_rmse=r.get('episode',{}).get('velocity_tracking_rmse'),mean_abs_mu=r['policy_mean_abs']))
    df=pd.DataFrame(rows)
    df['velocity_rmse']=pd.to_numeric(df['velocity_rmse'],errors='coerce')
    assert len(df)==30000 and df.iteration.tolist()==list(range(30000))
    output.mkdir(parents=True,exist_ok=True);df.to_csv(output/'gpt-训练趋势原始数据.csv',index=False)
    windows=[]
    for a,b in [(0,5000),(15000,20000),(20000,25000),(25000,30000)]:
        r=df.iloc[a:b].drop(columns='iteration').mean().to_dict();r.update(start=a,end_exclusive=b);windows.append(r)
    pd.DataFrame(windows).to_csv(output/'gpt-预注册时间窗口.csv',index=False)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for axis,key,label in zip(axes,['policy_std','tail_gt5','saturation','velocity_rmse'],['策略平均标准差','原始采样动作绝对值 >5 的比例','力矩包络饱和比例','训练已完成回合速度 RMSE（m/s）']):
        axis.plot(df.iteration,df[key],color='gray',alpha=.2,linewidth=.5)
        axis.plot(df.iteration,df[key].rolling(100,min_periods=1).mean(),color='tab:blue',linewidth=1)
        axis.axvline(20000,color='red',linestyle='--',alpha=.6)
        if stage==1 and not repaired_run:axis.axvline(300,color='gray',linestyle=':',alpha=.7)
        if not df[key].notna().any():
            axis.text(.5,.5,'训练日志未记录该指标；不以奖励代替',transform=axis.transAxes,ha='center',fontproperties=font)
        axis.set_ylabel(label,fontproperties=font);axis.grid(alpha=.2)
    zoom=axes[-1].inset_axes([.24,.34,.70,.56])
    zoom.plot(df.iteration,df.velocity_rmse,color='gray',alpha=.2,linewidth=.4)
    zoom.plot(df.iteration,df.velocity_rmse.rolling(100,min_periods=1).mean(),color='tab:blue',linewidth=.8)
    zoom.set_ylim(0,1);zoom.axvline(20000,color='red',linestyle='--',alpha=.6);zoom.grid(alpha=.2)
    zoom.set_title('0–1 m/s 放大；完整范围与离群值保留在主图',fontproperties=font,fontsize=9)
    axes[-1].set_xlabel('全局 PPO 更新索引；蓝线为 100 次更新窗口均值，红线为熵中点',fontproperties=font)
    title='正式训练描述性趋势；300 次处有工程中断，不是最终检查点评估' if stage==1 else 'Stage2 训练趋势；训练日志有遗漏交互警告，物理有效性存疑'
    if repaired_run:title='Stage%d 容量修复后训练趋势；未检测到原生物理错误，不是最终模型评估'%stage
    fig.suptitle(title,fontproperties=font)
    fig.tight_layout();fig.savefig(output/'gpt-四条训练趋势.png',dpi=160)
    print(pd.DataFrame(windows).to_json(orient='records'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',type=int,choices=(1,2),default=1);p.add_argument('--repaired-run',action='store_true');a=p.parse_args();run(a.input,a.output,a.stage,a.repaired_run)
