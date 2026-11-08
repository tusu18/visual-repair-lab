"""Small cross-platform determinism and actual rendering-cost receipt."""
import sys,json,hashlib,time,statistics,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_repair_lab.adaptive.env import AdaptiveRepairEnv
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.adaptive.probes import experiments
p=Path(__file__).resolve().parents[1]
bank=json.loads((p/'banks/train.json').read_text())['records'];rows=[];render=[]
for family in sorted({r['spec']['family'] for r in bank}):
 spec=next(r['spec'] for r in bank if r['spec']['family']==family);env=AdaptiveRepairEnv(spec,evidence='image');obs,_=env.reset(seed=7);hashes=[]
 actions=[{'type':'edit','source':compile_source(spec)},{'type':'run'}]+[{'type':'interact','event':e} for e in experiments(family)[0]]+[{'type':'submit'}]
 for action in actions:
  obs,reward,terminated,truncated,_=env.step(action)
  hashes.append({'context_sha256':hashlib.sha256(obs['context'].encode()).hexdigest(),'rgb_sha256':hashlib.sha256(obs['image'].tobytes()).hexdigest(),'reward':reward})
 for _ in range(10):
  start=time.perf_counter();env._refresh();render.append(time.perf_counter()-start)
 rows.append({'family':family,'task_id':spec['id'],'observations':hashes,'success':env.final_result['success']})
result={'passed':all(r['success'] for r in rows),'python':platform.python_version(),'platform':platform.platform(),'rows':rows,
        'render_refresh_median_ms':1000*statistics.median(render),'render_scope':'Two panels, reference computation and RGB concatenation; excludes PNG encoding, inference, HTTP and grading.'}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}));raise SystemExit(not result['passed'])
