"""Run against the local GUI + scripts/mock_model.py. Scripted, never VLM evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
 p=argparse.ArgumentParser();p.add_argument('--endpoint',default='http://127.0.0.1:8822');p.add_argument('--output',default='validation/http-smoke.json');a=p.parse_args()
 def request(path,body=None):
  data=None if body is None else json.dumps(body).encode()
  req=urllib.request.Request(a.endpoint+path,data=data,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
 tasks=request('/api/tasks')['tasks'];selected=[]
 for family in ('drag_hysteresis','linked_brush','contact_dynamics'):
  selected.append(next(t['id'] for t in tasks if t['family']==family))
 start=time.monotonic()
 created=request('/api/benchmark',dict(endpoint='http://127.0.0.1:8008/v1',model='scripted-submit-smoke',tasks=selected,repetitions=8,workers=8))
 deadline=time.monotonic()+120
 while True:
  report=request('/api/run?id='+created['run_id'])
  if not report['running']:break
  if time.monotonic()>deadline:raise TimeoutError('Smoke exceeded 120 seconds')
  time.sleep(.25)
 assert report['completed']==24 and report['interrupted']==0
 assert all(r['total_return']==-.025 for r in report['attempts'])
 # Eight simultaneous HTTP creations and steps exercise backlog and actual API.
 def attempt(i):
  r=request('/episodes',dict(task_id=selected[i%3]));eid=r['episode_id']
  payload=dict(episode_id=eid,request_id='same-request',expected_step=0,action={'type':'submit'})
  x=request('/step',payload);y=request('/step',payload)
  assert x==y and x['result']['step']==1 and x['result']['reward']==-.025
  assert 'checkpoint' not in x and 'rules' not in x['result']['context']
  return {'ok':True,'reward':x['result']['reward'],'step':1}
 with ThreadPoolExecutor(8) as pool:parallel=list(pool.map(attempt,range(8)))
 result={'passed':True,'scripted_only':True,'model_inference':False,'benchmark':report,
  'parallel_http':parallel,'seconds':time.monotonic()-start,'health':request('/health')}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
 print(json.dumps({'passed':True,'completed_scripted_attempts':24,'parallel_http':len(parallel),'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
