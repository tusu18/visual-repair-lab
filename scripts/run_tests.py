"""Bounded full-suite runner; preserves failures/timeouts instead of claiming pass."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=420)
args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
start=time.monotonic();timed_out=False
with (root/'pytest.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-m','pytest','-n','2','-q','--disable-warnings'],
        cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
    try:code=process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
        else:process.kill()
        code=process.wait()
receipt={'passed':code==0 and not timed_out,'returncode':code,'timed_out':timed_out,
         'deadline_seconds':args.timeout,'workers':2,'wall_seconds':time.monotonic()-start,
         'python':platform.python_version(),'platform':platform.platform()}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
raise SystemExit(0 if receipt['passed'] else 1)
