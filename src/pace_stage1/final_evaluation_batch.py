"""Isolated simulator processes with native-log verification for final evaluation."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def run(output,video=False,group=None,checkpoint=None):
    with (ROOT/'provenance/gpt-Stage1-PACEv2-最终评估样本计划-v1.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    keys=list(dict.fromkeys((r['terrain'],r['difficulty_row']) for r in rows))
    for i,(terrain,row) in enumerate(keys):
        if group is not None and i!=group:
            continue
        if video and row not in ('2','5','8') and terrain!='flat_reference':
            continue
        directory=output/('gpt-%02d-%s-%s'%(i,terrain,row))
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'gpt-调度锁').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            marker=directory/'gpt-完成记录.json'
            if marker.exists() and json.loads(marker.read_text()).get('runtime_log_checked'):
                if checkpoint is not None and Path(json.loads(marker.read_text())['checkpoint']).resolve()!=checkpoint.resolve():
                    raise RuntimeError('output already contains a different policy checkpoint')
                continue
            log=directory/'gpt-原生运行日志.log'
            command=[sys.executable,'-m','pace_stage1.final_evaluation','--output',str(output),'--group',str(i)]
            if checkpoint is not None:
                command.extend(['--checkpoint',str(checkpoint)])
            if video:
                command.append('--video')
            if log.exists():
                log.rename(directory/('gpt-原生运行日志-历史-%d.log'%time.time_ns()))
            with log.open('w') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            content=log.read_text(errors='replace')
            invalid=re.search(r'will miss interactions|invalid parameter|Fatal Python error|CUDA error|out of memory',content,re.I)
            if result.returncode or invalid or not marker.exists():
                (directory/'gpt-运行审核失败.json').write_text(json.dumps(dict(returncode=result.returncode,native_error=invalid.group(0) if invalid else None,completion_file_present=marker.exists()),indent=2)+'\n')
                if marker.exists():
                    marker.rename(directory/'gpt-无效完成记录.json')
                raise RuntimeError('evaluation group %d failed native runtime audit; see %s'%(i,log))
            data=json.loads(marker.read_text());data['runtime_log_checked']=True
            temporary=marker.with_suffix('.tmp')
            temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
            os.replace(temporary,marker)
            print('validated group',i,terrain,row,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--group',type=int)
    parser.add_argument('--checkpoint',type=Path,help='Explicit Stage2/3 final checkpoint; omitted retains Stage1 default')
    args=parser.parse_args()
    run(args.output,args.video,args.group,args.checkpoint)
