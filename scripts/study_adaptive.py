"""Prospective fixed-vs-adaptive sampling pilot, without policy weight updates."""
import argparse,concurrent.futures,json,random,time
from pathlib import Path
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.adaptive.curriculum import Curriculum,signal_summary
from visual_repair_lab.adaptive.generator import edit_task,evaluate_proposal
from visual_repair_lab.adaptive.scripted import run_attempt
from visual_repair_lab.adaptive.dsl import seal
from vision_sort.visual_training.tasks import oracle_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--rounds',type=int,default=4);p.add_argument('--groups-per-round',type=int,default=12);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);bank=json.loads(Path(a.pool).read_text())
    records=bank['records'];prior={f:[] for f in ('drag_hysteresis','linked_brush','contact_dynamics')}
    for r in records:
        f=r['spec']['family'];source=compile_source(r['spec'])
        if source not in prior[f]:prior[f].append(source)
    for f in prior:
        prior[f]+=[oracle_source(f,m,'train') for m in (0,64,128,256)]
    (out/'frozen-student-prior.json').write_text(json.dumps(prior,indent=2))
    curriculum=Curriculum(out/'curriculum.db','catalogue-proxy-v1');rng=random.Random(1707)
    for r in records:curriculum.add(r)
    groups=[];edits=[];serial=0;started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        def group(record,arm,round_index,index):
            spec=record['spec'];jobs=[]
            for j in range(8):
                aid=f'{arm}-{round_index}-{index}-{j}'
                jobs.append((spec,prior[spec['family']],700000+round_index*100000+index*8+j,aid))
            attempts=list(pool.map(run_attempt,jobs))
            folder=out/'attempts';folder.mkdir(exist_ok=True)
            summaries=[]
            for result in attempts:
                (folder/(result['attempt_id']+'.json')).write_text(json.dumps(result,indent=2))
                summaries.append({k:v for k,v in result.items() if k!='trace'})
            summary=curriculum.record(spec['id'],round_index,summaries) if arm!='fixed' else __import__('visual_repair_lab.adaptive.curriculum',fromlist=['summarize_group']).summarize_group(summaries)
            row={'arm':arm,'round':round_index,'task_id':spec['id'],'budget':spec['budget'],'summary':summary,'attempts':summaries}
            groups.append(row);(out/'groups.json').write_text(json.dumps(groups,indent=2));return row
        for i,r in enumerate(records):
            group(r,'calibration',0,i)
            if (i+1)%10==0:print('calibration',i+1,len(records),flush=True)
        for round_index in range(1,a.rounds+1):
            for index in range(a.groups_per_round):
                selected,selection=curriculum.choose()
                if selected is None:break
                row=group(selected,'adaptive',round_index,index);row['selection']=selection
                fixed=rng.choice(records);group(fixed,'fixed',round_index,index)
                child=edit_task(selected['spec'],row['summary']['p'],rng,serial)
                if child:
                    serial+=1;spec,lineage=child;record=evaluate_proposal(spec)
                    edits.append({'record':record,'lineage':lineage})
                    if record['certificate']['accepted']:curriculum.add(record,lineage)
                    (out/'edits.json').write_text(json.dumps(edits,indent=2))
            print('round',round_index,flush=True)
            by_arm={arm:signal_summary([g for g in groups if g['arm']==arm and g['round']>0]) for arm in ('fixed','adaptive')}
            (out/'summary.json').write_text(json.dumps({'arms':by_arm,'rounds':[
                {'round':r,'arm':arm,**signal_summary([g for g in groups if g['arm']==arm and g['round']==r])}
                for r in range(1,round_index+1) for arm in ('fixed','adaptive')],
                'calibration':signal_summary([g for g in groups if g['arm']=='calibration']),
                'wall_seconds':time.monotonic()-started,'edits_proposed':len(edits),'edits_accepted':sum(e['record']['certificate']['accepted'] for e in edits),
                'policy':'Fixed train-program catalogue, numeric observations, no policy weight updates. Not a VLM vision result.'},indent=2))

if __name__=='__main__':main()
