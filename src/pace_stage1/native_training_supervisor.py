"""Explicit repaired-run launcher; native physics errors stop the owned child."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ERROR=re.compile(r'will miss interactions|invalid parameter|CUDA error|out of memory|Fatal Python error',re.I)


def supervise(command,log_path,status_path):
    started=time.time();failure=None
    with log_path.open('x') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush()
            if failure is None and ERROR.search(line):
                failure=dict(elapsed_seconds=time.time()-started,message=line.strip())
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill()
        code=process.wait()
        process.stdout.close()
    result=dict(state='PROCESS_COMPLETED' if code==0 and failure is None else 'FAILED',
                returncode=code,native_error=failure,seconds=time.time()-started)
    status_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def run(stage,output):
    if output.exists():raise ValueError('repaired experiment must use a new output directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    module='pace_stage1.pace_v2_formal' if stage==1 else 'pace_stage1.stage2_formal'
    # stdbuf is loaded before Isaac Gym so native C stdout is line-buffered too.
    command=['stdbuf','-oL','-eL',sys.executable,'-u','-m',module,'--output',str(output),'--construction-capacity-repair']
    if stage==2:command+=['--potential-sign','-1']
    log=output.parent/(output.name+'-原生监督.log')
    status=output.parent/(output.name+'-原生监督.json')
    result=supervise(command,log,status)
    if result['state']=='FAILED':raise RuntimeError('training stopped; see '+str(status))
    manifest=json.loads((output/'gpt-运行记录.json').read_text())
    if manifest.get('state')!='completed' or manifest.get('completed_iterations')!=30000:
        raise RuntimeError('child exited without completing the formal run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.stage,a.output)
