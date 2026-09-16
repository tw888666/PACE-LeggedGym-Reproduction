"""Stratified final-evaluation tables, exact action tails, and evaluation-only uncertainty."""
import argparse,json,csv
from pathlib import Path
import numpy as np

METRICS=['velocity_rmse','yaw_rmse','survival','joint_task_success','duration_s','mean_abs_mu','saturation','eq9_activation','hard_limit_exceedance','electrical_w','mechanical_positive_net_w','diagnostic_power_w','prob_abs_mu_gt_2','prob_abs_mu_gt_5','prob_abs_mu_gt_10','crossed_tile']

def tails(values):
    values=np.asarray(values)
    return dict(zip(('p95','p99','p999','p9999'),np.quantile(values,[.95,.99,.999,.9999]).tolist()),scalar_count=int(values.size),prob_gt2=float((values>2).mean()),prob_gt5=float((values>5).mean()),prob_gt10=float((values>10).mean()),maximum=float(values.max()))


def run(root):
    import pandas as pd
    groups=sorted((root/'quantitative').glob('gpt-*/gpt-完成记录.json'))
    if len(groups)!=51:raise RuntimeError('all 51 planned groups must complete before final report')
    entry_path=root/'gpt-评估入口.json'
    entry=json.loads(entry_path.read_text()) if entry_path.exists() else {}
    checkpoints={str(Path(json.loads(p.read_text())['checkpoint']).resolve()) for p in groups}
    if len(checkpoints)!=1:raise RuntimeError('evaluation mixes different policy checkpoints')
    if entry.get('checkpoint') and checkpoints!={str(Path(entry['checkpoint']).resolve())}:
        raise RuntimeError('evaluation checkpoint differs from declared entry')
    rows=[];tail_rows=[];family_values={}
    for marker in groups:
        directory=marker.parent
        records=json.loads((directory/'gpt-逐回合指标.json').read_text());rows.extend(records)
        with np.load(directory/'gpt-原始动作样本.npz') as data:raw=np.abs(data['actions'])
        for seed in sorted({r['seed'] for r in records}):
            for command in ('standing','low_speed','normal_motion'):
                indices=[i for i,r in enumerate(records) if r['seed']==seed and r['command_bin']==command]
                selected=raw[:,indices,:];values=selected[np.isfinite(selected)]
                tail_rows.append(dict(terrain=records[0]['terrain'],difficulty_row=records[0]['difficulty_row'],seed=seed,command_bin=command,**tails(values)))
        family=records[0]['terrain']
        family_values.setdefault(family,[]).append(raw[np.isfinite(raw)])
    frame=pd.DataFrame(rows)
    if len(frame)!=29376 or frame.case_id.nunique()!=29376:raise RuntimeError('case coverage mismatch')
    plan_path=Path(__file__).resolve().parents[2]/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv'
    plan={r['case_id']:r for r in csv.DictReader(plan_path.open())}
    for row in rows:
        if row['case_id'] not in plan or any(str(row[k])!=v for k,v in plan[row['case_id']].items()):
            raise RuntimeError('observed case metadata differs from preregistered plan')
    for marker in groups:
        data=json.loads(marker.read_text())
        if data.get('state')!='completed' or not data.get('normalization_frozen') or not data.get('runtime_log_checked'):
            raise RuntimeError('incomplete or unfrozen normalization record')
        if Path(data['checkpoint']).name!='gpt-model-30000.pt':
            raise RuntimeError('unexpected checkpoint selection')
    frame['difficulty_row']=pd.to_numeric(frame.difficulty_row,errors='coerce').fillna(-1).astype(int)
    frame['seed']=frame.seed.astype(int)
    counts=frame.groupby(['terrain','difficulty_row','seed','command_bin']).size()
    if len(counts)!=459 or not (counts==64).all():raise RuntimeError('stratum allocation mismatch')
    output=root/'gpt-最终报告';output.mkdir(exist_ok=True)
    frame.to_csv(output/'gpt-逐回合结果.csv',index=False)
    from pace_stage0.constants import CANONICAL_JOINT_NAMES
    joint_rows=[]
    for family,subset in frame.groupby('terrain'):
        matrix=np.asarray(subset.per_joint_saturation.tolist(),dtype=float)
        for name,value in zip(CANONICAL_JOINT_NAMES,matrix.mean(0)):
            joint_rows.append(dict(terrain=family,joint=name,saturation=float(value)))
    pd.DataFrame(joint_rows).to_csv(output/'gpt-地形逐关节饱和.csv',index=False)
    pd.DataFrame(tail_rows).to_csv(output/'gpt-逐分层精确动作尾部.csv',index=False)
    primary=frame[frame.terrain!='flat_reference']
    for filename,by in [('地形',['terrain']),('地形难度',['terrain','difficulty_row']),('地形指令',['terrain','command_bin']),('地形难度指令',['terrain','difficulty_row','command_bin']),('种子',['seed']),('指令',['command_bin'])]:
        data=primary if filename in ('种子','指令') else frame
        data.groupby(by)[METRICS].mean().to_csv(output/('gpt-%s统计.csv'%filename))
    tail_family=[];all_primary=[]
    for family,chunks in family_values.items():
        values=np.concatenate(chunks)
        tail_family.append(dict(terrain=family,**tails(values)))
        if family!='flat_reference':all_primary.append(values)
    tail_family.append(dict(terrain='primary_all_surviving_action_scalars',**tails(np.concatenate(all_primary))))
    pd.DataFrame(tail_family).to_csv(output/'gpt-地形合并动作尾部.csv',index=False)
    del family_values,all_primary
    # Stratified bootstrap: resample 64 cases within each seed/terrain/row/command cell.
    rng=np.random.default_rng(910001);boot=[];labels=[]
    for key,group in frame.groupby(['terrain','difficulty_row','seed','command_bin'],sort=True):
        values=group[METRICS].to_numpy(dtype=float)
        indices=rng.integers(0,len(values),size=(2000,len(values)))
        boot.append(values[indices].mean(1));labels.append(key)
    boot=np.stack(boot)
    cis=[]
    for family in ['primary']+sorted(frame.terrain.unique().tolist()):
        selected=[i for i,k in enumerate(labels) if (k[0]!='flat_reference' if family=='primary' else k[0]==family)]
        distribution=boot[selected].mean(0)
        bounds=np.quantile(distribution,[.025,.975],axis=0)
        for j,metric in enumerate(METRICS):cis.append(dict(group=family,metric=metric,lower=float(bounds[0,j]),upper=float(bounds[1,j])))
    pd.DataFrame(cis).to_csv(output/'gpt-评估随机性置信区间.csv',index=False)
    means=primary[METRICS].mean()
    success=bool(means.joint_task_success>=.8 and means.survival>=.9)
    summary=dict(case_count=len(frame),primary_case_count=len(primary),metrics={k:float(v) for k,v in means.items()},task_adequate=success,task_classification='A_OR_B_REQUIRES_FIXED_VIDEO_REVIEW' if success else 'C',training_seeds=1,evaluation_seeds=3,checkpoint_std=json.loads(groups[0].read_text())['checkpoint_std'],uncertainty='evaluation randomness only; not training replication')
    if entry.get('training_validity'):
        summary['training_validity']=entry['training_validity']
        if entry['training_validity'].startswith('CONCERN'):
            summary['task_classification']='NUMERICAL_TASK_PASS_WITH_TRAINING_VALIDITY_CONCERN' if success else 'NUMERICAL_TASK_FAIL_WITH_TRAINING_VALIDITY_CONCERN'
    (output/'gpt-最终汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # Static scientific figures; separate final deterministic results from training curves.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    names={'smooth_slope':'光滑坡','rough_slope':'粗糙坡','stairs_negative_step':'负台阶楼梯','stairs_positive_step':'正台阶楼梯','boxes':'离散障碍','flat_reference':'平地参照'}
    terrain=frame.groupby('terrain')[METRICS].mean()
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for axis,metric,title in zip(axes.flat,['velocity_rmse','joint_task_success','saturation','diagnostic_power_w'],['线速度 RMSE（均方根误差，m/s）','联合任务成功率','力矩包络饱和比例','诊断功率和（W）']):
        axis.bar(range(len(terrain)),terrain[metric]);axis.set_title(title,fontproperties=font)
        axis.set_xticks(range(len(terrain)));axis.set_xticklabels([names[k] for k in terrain.index],fontproperties=font,rotation=25)
        axis.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'gpt-最终地形分层.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
