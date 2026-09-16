"""Finish the authorized evaluation report automatically after audited data arrive."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def count(root,mode):
    return sum(bool(json.loads(p.read_text()).get('runtime_log_checked'))
               for p in (root/mode).glob('*/gpt-完成记录.json'))


def run(root,stage=1):
    manifest=root/'gpt-评估总体进度.json'
    def update(state):
        data=dict(state=state,quantitative_groups=count(root,'quantitative'),
                  quantitative_groups_required=51,video_groups=count(root,'videos'),
                  video_groups_required=16,updated_unix=time.time())
        temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        os.replace(temporary,manifest)
    try:
        while count(root,'quantitative')<51:
            update('QUANTITATIVE_RUNNING');time.sleep(30)
        subprocess.run([sys.executable,'-m','pace_stage1.final_evaluation_report','--root',str(root)],check=True)
        while count(root,'videos')<16:
            update('QUANTITATIVE_REPORTED_VIDEO_RUNNING');time.sleep(30)
        subprocess.run([sys.executable,'-m','pace_stage1.final_evaluation_finish','--root',str(root),'--stage',str(stage)],check=True)
        update('COMPLETED')
    except Exception:
        update('REPORT_FAILED')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',type=int,choices=(1,2),default=1)
    args=parser.parse_args();run(args.root,args.stage)
