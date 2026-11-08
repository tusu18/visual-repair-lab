"""Private balanced benchmark sampling after a frozen source snapshot."""
import argparse,copy,json,random,secrets,time
from pathlib import Path
from visual_repair_lab.adaptive.generator import evaluate_proposal
from visual_repair_lab.adaptive.dsl import seal,HOOKS,VALUES,CONDITIONS,BASE_MASKS
from visual_repair_lab.adaptive.pool_gate import certify_against_catalogue
from visual_repair_lab.adaptive.private_checks import certify_private


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--catalogue',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed=secrets.randbits(64);rng=random.Random(seed);serial=0;records=[];rejected=[];labels={};started=time.monotonic()
 (out/'sampling-seed.json').write_text(json.dumps({'private_sampling_seed':seed,'source_freeze_precedes_sampling':True}));(out/'sampling-seed.json').chmod(0o600)
 previous=json.loads(Path(a.catalogue).read_text())['records'];seen={r['certificate']['behavior_fingerprint'] for r in previous}
 families=('drag_hysteresis','linked_brush','contact_dynamics');levels=(('easy',1,48,'baseline'),('medium',2,32,'baseline'),('hard',3,24,'starter'))
 def record_rejection(spec,reasons):
  rejected.append({'spec':spec,'reasons':reasons});(out/'rejection-prefix.json').write_text(json.dumps(rejected,indent=2))
 def sample(family,level,k,budget,start,index):
  nonlocal serial
  for _ in range(2500):
   serial+=1
   if serial>=10000:raise RuntimeError('Private proposal seed namespace exhausted; no model task calls permitted')
   ops=rng.sample(list(HOOKS[family]),k)
   rules=[dict(op=op,hook=HOOKS[family][op],condition=rng.choice(CONDITIONS[family]),value=rng.choice(VALUES[family][op])) for op in ops]
   spec=dict(id=f'eval-{family}-{level}-{index:02}',family=family,world={'mask':rng.choice(BASE_MASKS)},rules=rules,budget=budget,
             starting_point=start,split='development',scene_seed=3490000+serial)
   row=evaluate_proposal(spec)
   if row['certificate']['accepted'] and row['certificate']['behavior_fingerprint'] in seen:
    row['certificate']['accepted']=False;row['certificate']['reasons'].append('previously_used_behavior')
   if row['certificate']['accepted']:
    gate=certify_against_catalogue(row,previous+records+[row]);row['certificate']['whole_catalogue_gate']=gate
    if not gate['accepted']:row['certificate']['accepted']=False;row['certificate']['reasons']+=gate['reasons']
   if not row['certificate']['accepted']:record_rejection(spec,row['certificate']['reasons']);continue
   seen.add(row['certificate']['behavior_fingerprint']);return row
  raise RuntimeError('Bounded proposal search exhausted; rejection prefix preserved')
 for family in families:
  for level,k,budget,start in levels:
   for i in range(10):
    row=sample(family,level,k,budget,start,i);records.append(row);labels[row['spec']['id']]={'family':family,'level':level,'scope':'known_family_new_composition'}
    (out/'accepted-prefix.json').write_text(json.dumps(records,indent=2));print('ACCEPT',row['spec']['id'],flush=True)
 # Adding hypotheses can alter the common greedy tree. Recheck every target
 # against the completed corpus; replace only eligibility failures, never model
 # failures. Stop boundedly if no complete valid cohort can be built.
 for qualification_round in range(10):
  catalogue=previous+records;bad=[]
  for row in records:
   gate=certify_against_catalogue(row,catalogue);row['certificate']['whole_catalogue_gate']=gate
   if not gate['accepted']:bad.append(row)
  if not bad:break
  for row in bad:
   records.remove(row);spec=row['spec'];record_rejection(spec,['completed_corpus:'+x for x in row['certificate']['whole_catalogue_gate']['reasons']])
   level=labels[spec['id']]['level'];_,k,budget,start=next(x for x in levels if x[0]==level)
   records.append(sample(spec['family'],level,k,budget,start,int(spec['id'][-2:])))
  print('REQUALIFY',qualification_round,len(bad),flush=True)
 else:raise RuntimeError('Global qualification did not converge; no model task calls permitted')
 catalogue=previous+records;cv=[]
 for level,_,budget,_ in levels:
  for i in range(10):
   for _ in range(100):
    settings=dict(xywh=rng.choice((False,True)),center=rng.choice((False,True)),confidence=rng.choice((.45,.65)),nms=rng.choice((.35,.65)),swap=rng.choice((False,True)),clip=rng.choice((False,True)))
    spec=dict(id=f'eval-detection_overlay-{level}-{i:02}',family='detection_overlay',split='private_test',budget=budget,scene_seed=3516500+len(cv),grading_seed=rng.getrandbits(64),pipeline=settings)
    certificate=certify_private(spec)
    if certificate['accepted']:break
    record_rejection(spec,certificate['reasons'])
   else:raise RuntimeError('Private authority exhausted acceptance attempts')
   cv.append({'spec':spec,'certificate':certificate});labels[spec['id']]={'family':'detection_overlay','level':level,'scope':'held_out_family'};print('ACCEPT',spec['id'],flush=True)
 bank={'version':'6.0.0-final-benchmark','records':records+cv};bank['bank_sha256']=seal(bank['records'])
 (out/'bank.json').write_text(json.dumps(bank,indent=2));(out/'catalogue.json').write_text(json.dumps({'records':catalogue},indent=2))
 (out/'rejections.json').write_text(json.dumps(rejected,indent=2));(out/'labels.json').write_text(json.dumps(labels,indent=2))
 summary={'tasks':len(bank['records']),'families':{f:sum(r['spec']['family']==f for r in bank['records']) for f in families+('detection_overlay',)},'levels':{level:sum(labels[r['spec']['id']]['level']==level for r in bank['records']) for level,_,_,_ in levels},'rejected':len(rejected),'bank_sha256':bank['bank_sha256'],'wall_seconds':time.monotonic()-started,'actor_task_calls':0,'source_freeze_precedes_sampling':True}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 for f in out.iterdir():f.chmod(0o600)
if __name__=='__main__':main()
