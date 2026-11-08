"""Execute each released ideal policy using public numeric observations only.

The trusted validation prior is the full declared train program catalogue. It is
an upper bound, not an actor/VLM baseline or an arbitrary-program proof.
"""
import argparse,json,time,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_repair_lab.adaptive.env import AdaptiveRepairEnv
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.adaptive.certify import visible_signature

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--catalogue',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 bank=json.loads(Path(a.bank).read_text())['records'];catalogue=json.loads(Path(a.catalogue).read_text())['records']
 rows=[];start=time.monotonic();render=[]
 for record in bank:
  spec=record['spec'];gate=record['certificate']['whole_catalogue_gate'];tree=gate['private_tree'];env=AdaptiveRepairEnv(spec,evidence='numeric_text');obs,_=env.reset()
  while 'leaf' not in tree:
   seq=gate['private_sequences'][tree['experiment']]
   for event in seq:obs,*_=env.step({'type':'interact','event':event})
   context=json.loads(obs['context']);signature=visible_signature(context['displayed_values']['reference'])
   tree=tree['branches'][signature]
  sources=[compile_source(r['spec']) for r in catalogue if r['spec']['family']==spec['family']]
  source=sources[gate['private_representatives'][tree['leaf']]]
  for action in ({'type':'edit','source':source},{'type':'run'},{'type':'submit'}):env.step(action)
  t=time.monotonic();env._observation();render.append(time.monotonic()-t)
  rows.append({'task_id':spec['id'],'success':env.final_result['success'],'actions':env.steps,'spare_actions':spec['budget']-env.steps})
 elapsed=time.monotonic()-start
 result={'passed':all(r['success'] and r['spare_actions']>=2 for r in rows),'tasks':len(rows),'rows':rows,'wall_seconds':elapsed,
         'episodes_per_second':len(rows)/elapsed,'observation_packaging_median_ms':1000*statistics.median(render),
         'scope':'Trusted finite-train-catalogue ideal policy, numeric observations; includes Python/evaluator/rendering, excludes model inference and HTTP latency.'}
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
 raise SystemExit(not result['passed'])
if __name__=='__main__':main()
