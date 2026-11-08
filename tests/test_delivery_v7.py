import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pytest
from visual_repair_lab.adaptive.dsl import HOOKS,VALUES,seal
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.v7.env import make_env,terminal_reward,leaf_agreement
from visual_repair_lab.v7.snapshot import dump,restore
from visual_repair_lab.v7.server import Store,Conflict
from visual_repair_lab.v7.benchmark import model_url,decode_call


def spec(family='drag_hysteresis'):
 return dict(id=family,family=family,world={'mask':0},rules=[dict(op='offset',hook=HOOKS[family]['offset'],condition='always',value=VALUES[family]['offset'][0])],budget=32,starting_point='baseline',split='train',scene_seed=3478000)


def bank(s=None):
 s=s or spec();rows=[{'spec':s,'certificate':{'accepted':True,'spec_sha256':seal(s)}}]
 return {'records':rows,'bank_sha256':seal(rows)}


@pytest.mark.parametrize('start,final,total,want',[(10,10,20,0),(10,15,20,4),(10,20,20,10),(10,5,20,-.5),(20,20,20,0),(0,10,20,4)])
def test_relative_terminal_formula(start,final,total,want):
 assert terminal_reward(start,final,total)==want


@pytest.mark.parametrize('family',['drag_hysteresis','linked_brush','contact_dynamics'])
@pytest.mark.parametrize('mode',['evaluation','train'])
def test_unchanged_and_comments_do_not_pay(family,mode):
 e=make_env(spec(family),reward_mode=mode);e.reset()
 e.step({'type':'edit','source':e.source+'\n# no change\n'});e.step({'type':'run'})
 _,r,_,_,_=e.step({'type':'submit'})
 assert e.reward_ledger[-1]['components']['terminal']==0
 assert r==pytest.approx(-.025)
 assert e.task_return+e.process_return<=0


@pytest.mark.parametrize('family',['drag_hysteresis','linked_brush','contact_dynamics'])
def test_repair_earns_improvement_and_preserves_grader(family):
 s=spec(family);e=make_env(s,reward_mode='train');e.reset()
 e.step({'type':'edit','source':compile_source(s)})
 assert e.reward_ledger[-1]['components']['repair_progress']>0
 e.step({'type':'run'});e.step({'type':'submit'})
 assert e.final_result['success'] and e.reward_ledger[-1]['components']['terminal']==10
 assert e.reward_ledger[-1]['components']['unactivated_submission']==0


@pytest.mark.parametrize('family',['drag_hysteresis','linked_brush','contact_dynamics'])
def test_snapshots_exact_images_rewards_rng_and_no_reset(family,monkeypatch):
 s=spec(family);config=dict(spec=s,mode='train',evidence='image');e=make_env(s,reward_mode='train');e.reset(seed=7)
 actions=[{'type':'edit','source':compile_source(s)},{'type':'run'},{'type':'interact','event':{'kind':'reset_scene'}},{'type':'submit'}]
 for a in actions:
  checkpoint=dump(e)
  f=restore(config,checkpoint)
  def forbidden(*args,**kwargs):raise AssertionError('Restore must never reset or replay')
  monkeypatch.setattr(f,'reset',forbidden)
  x=e.step(a);y=f.step(a)
  assert x[0]['context']==y[0]['context'] and np.array_equal(x[0]['image'],y[0]['image'])
  assert x[1:4]==y[1:4] and e.reward_ledger==f.reward_ledger
  assert e.np_random.bit_generator.state==f.np_random.bit_generator.state


def test_private_detection_checkpoint_and_relative_rewards():
 s=dict(id='detection-dev',family='detection_overlay',split='private_test',budget=24,scene_seed=3510100,pipeline=dict(xywh=True,center=True,confidence=.45,nms=.35,swap=False,clip=True))
 e=make_env(s);e.reset();f=restore(dict(spec=s,mode='evaluation',evidence='image'),dump(e))
 x=e.step({'type':'submit'});y=f.step({'type':'submit'})
 assert x[1]==y[1]==-.025 and x[0]['context']==y[0]['context']


def test_corrupt_snapshot_rejected():
 s=spec();e=make_env(s);e.reset();snapshot=dump(e);snapshot['sha256']='0'*64
 with pytest.raises(ValueError):restore(dict(spec=s,mode='evaluation',evidence='image'),snapshot)


def test_checkpoint_worker_restart_timeout_and_public_boundary(tmp_path,monkeypatch):
 b=bank();store=Store(tmp_path/'episodes.db',b);created=store.create(spec()['id']);eid=created['episode_id']
 assert 'checkpoint' not in created
 first=store.step(eid,'saved',0,{'type':'observe'})
 reopened=Store(tmp_path/'episodes.db',b)
 assert reopened.get(eid)==first and reopened.step(eid,'saved',0,{'type':'observe'})==first
 with pytest.raises(Conflict):reopened.step(eid,'saved',0,{'type':'submit'})
 monkeypatch.setattr(store,'worker',lambda *args:dict(ok=False,error='evaluation_timeout',scored=False,retryable=True))
 assert not store.step(eid,'retryable',1,{'type':'observe'})['ok']
 assert store.get(eid)['result']['step']==1
 assert reopened.step(eid,'retryable',1,{'type':'observe'})['result']['step']==2
 wire=json.dumps(first)
 assert not any(k in first for k in ('checkpoint','spec','rules','checks'))
 assert not any(k in first['result']['context'] for k in ('rules','world','certificate','_initial_passed'))


def test_different_episodes_evaluate_outside_db_lock(tmp_path,monkeypatch):
 store=Store(tmp_path/'episodes.db',bank());ids=[store.create(spec()['id'])['episode_id'] for _ in range(2)]
 barrier=threading.Barrier(2);original=store.worker
 def worker(*args):
  barrier.wait(timeout=5)
  # Another independent writer can commit during both evaluations.
  with store.connect() as db:db.execute('INSERT INTO interruptions VALUES (?,?,?)',('probe',time.time(),'lock_probe'))
  return original(*args)
 monkeypatch.setattr(store,'worker',worker)
 with ThreadPoolExecutor(2) as pool:
  values=list(pool.map(lambda eid:store.step(eid,'a',0,{'type':'observe'}),ids))
 assert all(v['ok'] for v in values)


@pytest.mark.parametrize('same_id',[False,True])
def test_concurrent_same_episode_commits_once(tmp_path,monkeypatch,same_id):
 store=Store(tmp_path/'episodes.db',bank());eid=store.create(spec()['id'])['episode_id']
 barrier=threading.Barrier(2);original=store.worker
 def worker(*args):barrier.wait(timeout=5);return original(*args)
 monkeypatch.setattr(store,'worker',worker)
 def step(i):
  try:return store.step(eid,'same' if same_id else str(i),0,{'type':'observe'})
  except Conflict:return None
 with ThreadPoolExecutor(2) as pool:values=list(pool.map(step,range(2)))
 assert sum(v is not None for v in values)==(2 if same_id else 1)
 assert store.get(eid)['result']['step']==1
 with store.connect() as db:
  assert len(json.loads(db.execute('SELECT actions FROM episodes WHERE id=?',(eid,)).fetchone()[0]))==1


def test_field_progress_and_power_cap():
 assert leaf_agreement({'a':2,'b':7},{'a':2,'b':9})==.5
 e=make_env(spec(),reward_mode='train');e.reset();e.step({'type':'edit','source':compile_source(spec())});e.step({'type':'run'})
 e.step({'type':'interact','event':{'kind':'down','x':80,'y':250}})
 assert 0<e.reward_ledger[-1]['components']['test_power']<=.1
 e.step({'type':'edit','source':e.starter});e.step({'type':'submit'})
 assert e.reward_ledger[-1]['components']['terminal']==0
 assert e._positive['repair_progress']<=.75


def test_zero_whole_case_partial_repair_has_field_signal():
 s=next(r['spec'] for r in json.loads(Path('banks/train.json').read_text())['records'] if r['spec']['id']=='train-0001')
 e=make_env(s,reward_mode='train');e.reset();assert e._initial_passed==0
 source=compile_source(s).replace('def solve(state):','def _repair(state):')+'\ndef solve(state):\n    out = _repair(state)\n    out["projected"]["S1"][0][0] += 1\n    return out\n'
 e.step({'type':'edit','source':source})
 assert e._grade(source)['passed']==0
 assert e.reward_ledger[-1]['components']['repair_progress']>0


def test_model_adapter_no_remote_or_credential_urls():
 for endpoint in ('https://openrouter.ai/api/v1','http://user:key@localhost/v1','http://example.com/v1','file:///etc/passwd'):
  with pytest.raises(ValueError):model_url(endpoint)
 assert model_url('http://127.0.0.1:8000/v1').endswith('/v1/chat/completions')
 with pytest.raises(ValueError):decode_call({'response':{'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}})


def test_aged_priors_are_not_checkpoint_measurements(tmp_path):
 from visual_repair_lab.v7.curriculum import Curriculum
 from visual_repair_lab.adaptive.certify import certify
 from visual_repair_lab.adaptive.pool_gate import certify_against_catalogue
 s=spec();cert=certify(s);record={'spec':s,'certificate':cert};cert['whole_catalogue_gate']=certify_against_catalogue(record,[record])
 c=Curriculum(tmp_path/'c.db','old');c.add(record)
 rows=[dict(attempt_id=str(i),status='completed',success=i<4) for i in range(8)];c.record(s['id'],0,rows)
 new=Curriculum(tmp_path/'c.db','new');assert new.inherit('old',age=2)['inherited']==1
 row=new.snapshot()[s['id']];assert 'estimate' not in row and row['prior']['effective_attempts']==2
 assert row['prior']['p_prior']==.5 and row['prior']['scheduling_only']
 new.record(s['id'],0,rows);assert new.snapshot()[s['id']]['estimate']['n']==8


def test_gui_assets_packaged():
 text=Path('visual_repair_lab/v7/gui.html').read_text()
 assert 'Benchmark a model' in text and 'password' in text


def test_malformed_completed_call_is_preserved_on_resume(tmp_path,monkeypatch):
 from visual_repair_lab.v7.benchmark import Benchmarks
 store=Store(tmp_path/'episodes.db',bank());manager=Benchmarks(store,tmp_path/'runs');calls=[]
 def malformed(*args):
  calls.append(1)
  return {'response':{'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}]}}
 monkeypatch.setattr('visual_repair_lab.v7.benchmark.call_model',malformed)
 root=tmp_path/'attempts';root.mkdir();config={'tasks':[spec()['id']],'repetitions':1,'evidence':'image','endpoint':'http://localhost:8008/v1','model':'scripted-malformed'}
 manager.attempt(root,config,spec()['id'],0,'')
 saved=(root/'0000/000-call.json').read_bytes()
 manager.attempt(root,config,spec()['id'],0,'')
 assert len(calls)==1 and (root/'0000/000-call.json').read_bytes()==saved
 assert json.loads((root/'0000/checkpoint.json').read_text())['status']=='interrupted'


def test_empty_benchmark_does_not_silently_launch_whole_bank(tmp_path):
 from visual_repair_lab.v7.benchmark import Benchmarks
 manager=Benchmarks(Store(tmp_path/'episodes.db',bank()),tmp_path/'runs')
 with pytest.raises(ValueError):manager.start({'tasks':[],'endpoint':'http://localhost:8000/v1','model':'local'})


def test_live_source_change_refuses_new_work(tmp_path,monkeypatch):
 store=Store(tmp_path/'episodes.db',bank())
 monkeypatch.setattr('visual_repair_lab.v7.server.runtime_seal',lambda:'changed')
 r=store.create(spec()['id'])
 assert not r['ok'] and r['error']=='runtime_changed' and r['scored'] is False
