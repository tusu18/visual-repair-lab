"""Offline delivery checks, no model inference or training. Run from source root."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import tempfile
import time
from visual_repair_lab.v7.env import make_env
from visual_repair_lab.v7.server import Store
from visual_repair_lab.adaptive.server import Store as LegacyStore


def unchanged(row):
 e=make_env(row['spec'],reward_mode='train');e.reset();_,reward,_,_,_=e.step({'type':'submit'})
 return {'task':row['spec']['id'],'initial_passed':e._initial_passed,'total':e._initial_total,
  'terminal':e.reward_ledger[-1]['components']['terminal'],'reward':reward,'final_passed':e.final_result['passed']}


def bench(store_type,bank,episodes,steps):
 with tempfile.TemporaryDirectory(prefix='vrl7-throughput-') as directory:
  store=store_type(Path(directory)/'episodes.db',bank,timeout=90)
  task=bank['records'][0]['spec']['id'];latencies=[]
  def episode(i):
   created=store.create(task)
   if not created['ok']:return {'ok':False,'error':created}
   eid=created['episode_id']
   for j in range(steps):
    start=time.monotonic();r=store.step(eid,f'request-{j}',j,{'type':'interact','event':{'kind':'reset_scene'}})
    latencies.append(time.monotonic()-start)
    if not r['ok']:return {'ok':False,'error':r}
   return {'ok':True,'steps':store.get(eid)['result']['step']}
  start=time.monotonic()
  with ThreadPoolExecutor(episodes) as pool:rows=list(pool.map(episode,range(episodes)))
  seconds=time.monotonic()-start;latencies.sort()
  return {'implementation':store_type.__module__,'episodes':episodes,'steps_per_episode':steps,
   'seconds':seconds,'accepted_steps':sum(r.get('steps',0) for r in rows),'errors':sum(not r['ok'] for r in rows),
   'step_median_seconds':statistics.median(latencies),'step_p95_seconds':latencies[min(len(latencies)-1,int(len(latencies)*.95))],
   'step_max_seconds':max(latencies),'steps_per_second':episodes*steps/seconds,'outcomes':rows}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='validation');p.add_argument('--skip-pool',action='store_true');p.add_argument('--legacy',action='store_true');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);bank=json.loads(Path('banks/train.json').read_text())
 if not a.skip_pool:
  start=time.monotonic()
  with ProcessPoolExecutor(4) as pool:rows=list(pool.map(unchanged,bank['records']))
  report={'tasks':len(rows),'positive_unchanged_rewards':sum(r['reward']>0 for r in rows),
    'nonzero_terminal':sum(r['terminal']!=0 for r in rows),'seconds':time.monotonic()-start,'rows':rows}
  (out/'unchanged-submit-audit.json').write_text(json.dumps(report,indent=2))
  assert report['positive_unchanged_rewards']==report['nonzero_terminal']==0
  print('Unchanged-submit audit:',len(rows),'passed',flush=True)
 timings=[]
 for kind in ([LegacyStore,Store] if a.legacy else [Store]):
  for episodes in (1,8):
   r=bench(kind,bank,episodes,8);timings.append(r)
   (out/'concurrency-benchmark.json').write_text(json.dumps({'runs':timings,'note':'Local CPU worker timings, includes process start and rendering; no model latency.'},indent=2))
   print({k:v for k,v in r.items() if k!='outcomes'},flush=True)
   assert r['errors']==0


if __name__=='__main__':main()
