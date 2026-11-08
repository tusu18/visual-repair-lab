import argparse,concurrent.futures,json,random,time
from pathlib import Path
from visual_repair_lab.adaptive.generator import propose,evaluate_proposal,filter_records
from visual_repair_lab.adaptive.dsl import seal
from visual_repair_lab.adaptive.pool_gate import certify_against_catalogue

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=200);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rng=random.Random(7)
 specs=[propose(rng,i) for i in range(a.count)];(out/'proposals.json').write_text(json.dumps(specs,indent=2))
 rows=[];start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  for row in pool.map(evaluate_proposal,specs):
   rows.append(row)
   if len(rows)%20==0:print(len(rows),sum(r['certificate']['accepted'] for r in rows),flush=True)
   (out/'unfiltered-records.json').write_text(json.dumps(rows,indent=2))
 result=filter_records(rows)
 provisional=list(result['records']);accepted=[]
 for record in provisional:
  gate=certify_against_catalogue(record,provisional)
  record['certificate']['whole_catalogue_gate']=gate
  if gate['accepted']:accepted.append(record)
  else:
   record['certificate']['accepted']=False
   record['certificate']['reasons']+=gate['reasons']
   result['rejected'].append(record)
 result['records']=accepted
 result['rejection_counts']={}
 for record in result['rejected']:
  for reason in record['certificate']['reasons']:
   result['rejection_counts'][reason]=result['rejection_counts'].get(reason,0)+1
 result.update(version='6.0.0',wall_seconds=time.monotonic()-start)
 result['bank_sha256']=seal(result['records']);(out/'pool.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'proposed':a.count,'accepted':len(result['records']),'rejections':result['rejection_counts']}),flush=True)
if __name__=='__main__':main()
