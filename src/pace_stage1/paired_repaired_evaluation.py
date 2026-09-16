"""Run the existing final protocol for the two repaired 30k models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'artifacts/final-evaluation/gpt-容量修复成对评估-v1'


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def progress(root):
    return {mode:sum(bool(json.loads(p.read_text()).get('runtime_log_checked'))
                     for p in (root/mode).glob('*/gpt-完成记录.json'))
            for mode in ('quantitative','videos')}


def generate_reports(roots,runs,analysis_python,log_path):
    with log_path.open('x') as log:
        for stage,root in roots.items():
            for module,args in (
                ('final_evaluation_report',['--root',str(root)]),
                ('final_training_curves',['--input',str(runs[stage]/'gpt-逐迭代指标.jsonl'),'--output',str(root/'gpt-训练趋势'),'--stage',str(stage),'--repaired-run']),
                ('final_evaluation_finish',['--root',str(root),'--stage',str(stage)])):
                subprocess.run([analysis_python,'-u','-m','pace_stage1.'+module]+args,stdout=log,stderr=subprocess.STDOUT,check=True)


def run(report_only=False,analysis_python=sys.executable):
    subprocess.run([analysis_python,'-c','import numpy,pandas,matplotlib'],check=True)
    roots={s:ROOT/('artifacts/final-evaluation/gpt-stage%d-pace-v2-capacity-r2-v1'%s) for s in (1,2)}
    runs={s:ROOT/('artifacts/ppo/gpt-pace-v2-formal-stage%d-seed0-capacity-r2'%s) for s in (1,2)}
    if report_only:
        status_path=CONTROL/'gpt-评估总体进度.json'
        status=json.loads(status_path.read_text())
        for root in roots.values():
            if progress(root)!={'quantitative':51,'videos':16}:
                raise RuntimeError('all audited evaluation groups must complete before report-only recovery')
        write(CONTROL/('gpt-报告恢复前状态-%d.json'%time.time_ns()),status)
        status.update(state='REPORTING',analysis_python=analysis_python,updated_unix=time.time())
        status.pop('error',None)
        write(status_path,status)
        try:
            generate_reports(roots,runs,analysis_python,CONTROL/('gpt-报告恢复-%d.log'%time.time_ns()))
            status.update(state='COMPLETED',finished_unix=time.time())
        except Exception as error:
            status.update(state='FAILED',error=repr(error))
            raise
        finally:
            status['updated_unix']=time.time();write(status_path,status)
        return
    for stage,training in runs.items():
        manifest=json.loads((training/'gpt-运行记录.json').read_text())
        audit=json.loads(training.with_name(training.name+'-原生监督.json').read_text())
        if manifest['state']!='completed' or manifest['completed_iterations']!=30000 or not manifest.get('physics_construction_repair'):
            raise RuntimeError('repaired 30k training required')
        if audit['returncode']!=0 or audit['native_error'] is not None:
            raise RuntimeError('training native audit not passed')
        if not (training/'gpt-model-30000.pt').is_file():raise RuntimeError('final checkpoint missing')
        if roots[stage].exists():raise RuntimeError('use a fresh evaluation output; refusing to mix runs')
    CONTROL.mkdir(parents=True,exist_ok=True)
    for stage,root in roots.items():
        root.mkdir()
        entry=dict(stage=stage,checkpoint=str(runs[stage]/'gpt-model-30000.pt'),
                   construction_capacity_repaired=True,
                   training_validity='NO_NATIVE_ERRORS_DETECTED_IN_SUPERVISED_TRAINING',
                   baseline_root=str(roots[1]) if stage==2 else None,
                   protocol='STAGE1_PACE_V2_FINAL_EVALUATION_V1',
                   case_plan=str(ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv'),
                   expected_cases=29376,expected_videos=48,started_unix=time.time())
        write(root/'gpt-评估入口.json',entry)
    jobs=[]
    status=dict(state='STARTING',started_unix=time.time(),roots={str(k):str(v) for k,v in roots.items()})
    def update(state):
        status.update(state=state,updated_unix=time.time(),progress={str(s):progress(r) for s,r in roots.items()},
                      jobs=[dict(stage=s,mode=m,gpu=g,pid=p.pid,returncode=p.poll()) for s,m,g,p,f in jobs])
        write(CONTROL/'gpt-评估总体进度.json',status)
    try:
        for stage,mode,gpu in ((1,'quantitative',2),(2,'quantitative',3),(1,'videos',0),(2,'videos',1)):
            log=(CONTROL/('gpt-Stage%d-%s-调度.log'%(stage,mode))).open('x')
            command=[sys.executable,'-u','-m','pace_stage1.final_evaluation_batch',
                     '--output',str(roots[stage]/mode),'--checkpoint',str(runs[stage]/'gpt-model-30000.pt')]
            if mode=='videos':command+=['--video']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            jobs.append((stage,mode,gpu,process,log))
        while any(p.poll() is None for _,_,_,p,_ in jobs):
            update('EVALUATION_RUNNING' if all(p.poll() in (None,0) for _,_,_,p,_ in jobs) else 'WORKER_FAILED_OTHER_JOBS_CONTINUING')
            time.sleep(10)
        if any(p.returncode!=0 for _,_,_,p,_ in jobs):raise RuntimeError('evaluation worker failed; inspect its native audit log')
        update('REPORTING')
        generate_reports(roots,runs,analysis_python,CONTROL/'gpt-报告生成.log')
        update('COMPLETED')
    except Exception as error:
        status['error']=repr(error)
        update('FAILED')
        raise
    finally:
        for _,_,_,_,log in jobs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--analysis-python',default=sys.executable)
    args=parser.parse_args();run(args.report_only,args.analysis_python)
