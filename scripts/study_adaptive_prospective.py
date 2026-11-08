"""Prospective picker comparison reusing documented train calibration groups.

The prior pilot is preserved. No prior actor decisions or outcomes are regenerated.
"""
import argparse,concurrent.futures,json,random,time
from pathlib import Path
from visual_repair_lab.adaptive.curriculum import Curriculum,summarize_group,signal_summary
from visual_repair_lab.adaptive.generator import edit_task,evaluate_proposal
from visual_repair_lab.adaptive.scripted import run_attempt
from visual_repair_lab.adaptive.pool_gate import certify_against_catalogue

def main():
 p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--pool',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=2707);p.add_argument('--attempt-seed-base',type=int,default=1700000);a=p.parse_args()
 prior_root=Path(a.calibration);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 prior=json.loads((prior_root/'frozen-student-prior.json').read_text());records=json.loads(Path(a.pool).read_text())['records']
 ids={r['spec']['id'] for r in records}
 calibration=[g for g in json.loads((prior_root/'groups.json').read_text()) if g['arm']=='calibration' and g['task_id'] in ids]
 c=Curriculum(out/'curriculum.db','catalogue-proxy-v1',seed=a.seed);rng=random.Random(a.seed)
 for r in records:c.add(r)
 for row in calibration:c.record(row['task_id'],0,row['attempts'])
 (out/'calibration-provenance.json').write_text(json.dumps({'source':str(prior_root),'groups':len(calibration),'reused_without_regeneration':True,'selection_rule':'80% active/replay, up to20% unseen when both exist','policy_weights_changed':False},indent=2))
 groups=[];edits=[];serial=5000;start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for round_index in range(1,5):
   for index in range(12):
    selected,selection=c.choose()
    if selected is None:break
    for arm,record in (('adaptive',selected),('fixed',rng.choice(records))):
     s=record['spec'];jobs=[(s,prior[s['family']],a.attempt_seed_base+round_index*10000+index*8+j,f'prospective-{arm}-{round_index}-{index}-{j}') for j in range(8)]
     attempts=list(pool.map(run_attempt,jobs));summaries=[];folder=out/'attempts';folder.mkdir(exist_ok=True)
     for result in attempts:
      (folder/(result['attempt_id']+'.json')).write_text(json.dumps(result,indent=2));summaries.append({k:v for k,v in result.items() if k!='trace'})
     summary=c.record(s['id'],round_index,summaries) if arm=='adaptive' else summarize_group(summaries)
     row={'arm':arm,'round':round_index,'task_id':s['id'],'budget':s['budget'],'summary':summary,'attempts':summaries}
     if arm=='adaptive':row['selection']=selection;adaptive_row=row
     groups.append(row);(out/'groups.json').write_text(json.dumps(groups,indent=2))
    child=edit_task(selected['spec'],adaptive_row['summary']['p'],rng,serial)
    if child:
     serial+=1;s,lineage=child;record=evaluate_proposal(s)
     if record['certificate']['accepted']:
      catalogue=[v for v in c.snapshot().values()]+[record]
      gate=certify_against_catalogue(record,catalogue);record['certificate']['whole_catalogue_gate']=gate
      if not gate['accepted']:record['certificate']['accepted']=False;record['certificate']['reasons']+=gate['reasons']
     edits.append({'record':record,'lineage':lineage})
     if record['certificate']['accepted']:c.add(record,lineage)
     (out/'edits.json').write_text(json.dumps(edits,indent=2))
   summary={'arms':{arm:signal_summary([g for g in groups if g['arm']==arm]) for arm in ('fixed','adaptive')},
     'rounds':[{'round':r,'arm':arm,**signal_summary([g for g in groups if g['arm']==arm and g['round']==r])} for r in range(1,round_index+1) for arm in ('fixed','adaptive')],
     'wall_seconds':time.monotonic()-start,'edits_proposed':len(edits),'edits_accepted':sum(e['record']['certificate']['accepted'] for e in edits),
     'note':'Prospective train sampling diagnostic after a documented picker change; same used pool and frozen scripted prior. No weight updates.'}
   (out/'summary.json').write_text(json.dumps(summary,indent=2));print('round',round_index,{arm:summary['arms'][arm]['raw_mixed_share'] for arm in ('fixed','adaptive')},flush=True)
if __name__=='__main__':main()
