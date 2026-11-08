import copy
import json
import pytest
from visual_repair_lab.v5.compare import equivalent
from visual_repair_lab.v5.env import GeneratedRepairEnv
from visual_repair_lab.v5.generator import certify, seal, sources, states_for, validate_spec, wilson
from visual_repair_lab.v5.server import Store


def spec(family='linked_brush'):
    rules = {'linked_brush': {'pan_limit': 20., 'column_a': True},
             'drag_hysteresis': {'canvas_clamp': True, 'wide_gain': .5},
             'contact_dynamics': {'dt_max': .04, 'gravity_limit': 600.}}
    return dict(id=family, family=family, split='train', level='hard', mask=0,
                scene_seed=3310100, rules=rules[family])


def bank(s):
    records = [{'spec': s, 'certificate': certify(s)}]
    return {'records': records, 'bank_sha256': seal(records)}


def test_tolerance_not_rounding():
    assert equivalent({'x': [42.05-1e-12]}, {'x': [42.05+1e-12]})
    assert not equivalent({'x': [42.011]}, {'x': [42.019]})
    assert not equivalent(True, 1)
    assert not equivalent(float('nan'), 0)
    assert not equivalent(['A'], ['B'])


@pytest.mark.parametrize('family', ['drag_hysteresis', 'linked_brush', 'contact_dynamics'])
def test_generated_world_certificate_and_oracle(family):
    s = spec(family)
    receipt = certify(s)
    assert receipt['accepted'], receipt
    assert all(r['killed_checks'] >= 2 for r in receipt['rule_witnesses'].values())
    env = GeneratedRepairEnv(s, reward_mode='train')
    env.reset()
    env.step({'type': 'edit', 'source': sources(s)[0]})
    assert env.reward_ledger[-1]['components']['repair_progress'] > 0
    env.step({'type': 'run'})
    _, _, done, truncated, _ = env.step({'type': 'submit'})
    assert done and not truncated and env.final_result['success']
    assert env.final_result['passed'] == env.final_result['total']


def test_progress_is_not_repeated_by_run_or_reverting():
    s = spec(); env = GeneratedRepairEnv(s, reward_mode='train'); env.reset()
    gold = sources(s)[0]
    env.step({'type':'edit', 'source':gold})
    earned = env._positive['repair_progress']
    env.step({'type':'run'})
    assert env.reward_ledger[-1]['components']['repair_progress'] == 0
    env.step({'type':'edit', 'source':env.starter})
    assert env.reward_ledger[-1]['components']['regression'] < 0
    env.step({'type':'edit', 'source':gold})
    assert env._positive['repair_progress'] == earned


def test_no_pre_edit_test_power_and_revision_binding():
    s = spec(); env = GeneratedRepairEnv(s, reward_mode='train'); env.reset()
    probe = {'type':'interact','event':{'kind':'set_transform','zoom':1.,'pan_x':160,'pan_y':0}}
    env.step(probe)
    assert env.reward_ledger[-1]['components']['test_power'] == 0
    env.step({'type':'edit','source':sources(s)[0]})
    env.step(probe)  # saved, not activated
    assert env.reward_ledger[-1]['components']['test_power'] == 0
    env.step({'type':'run'}); env.step(probe)
    assert env.reward_ledger[-1]['components']['test_power'] > 0
    env.step(probe)
    assert env.reward_ledger[-1]['components']['test_power'] == 0


def test_unactivated_submission_still_grades_but_costs():
    s = spec(); env = GeneratedRepairEnv(s); env.reset()
    env.step({'type':'edit','source':sources(s)[0]})
    _, reward, _, _, _ = env.step({'type':'submit'})
    assert env.final_result['success']
    assert reward == pytest.approx(9.725)
    assert not env.final_result['final_revision_activated']


def test_partial_credit_formula():
    env = GeneratedRepairEnv(spec()); env.reset()
    _, reward, _, _, _ = env.step({'type':'submit'})
    f = env.final_result
    assert not f['success']
    assert reward == pytest.approx(-.025-2+8*f['passed']/f['total'])


def test_invalid_rule_rejected():
    s = spec(); s['rules']['pan_limit'] = '__import__("os")'
    with pytest.raises(ValueError): validate_spec(s)


def test_unknown_fields_rejected():
    s = spec(); s['python'] = 'malicious'
    with pytest.raises(ValueError): validate_spec(s)


def test_generated_progress_final_state_disjoint():
    s = spec(); env = GeneratedRepairEnv(s); env.reset()
    from vision_sort.visual_causal.env import digest
    assert {digest(x) for x in env._progress_cases}.isdisjoint({digest(x) for x in states_for(s,'final')})


@pytest.mark.parametrize('evidence', ['image','numeric_text','image_no_footer','none'])
def test_observation_arms_and_private_fields(evidence):
    env = GeneratedRepairEnv(spec(), evidence=evidence); observation, _ = env.reset()
    context = json.loads(observation['context'])
    assert not {'spec','rules','mask','oracle_source','world_seed','rule_witnesses'} & context.keys()
    assert ('displayed_values' in context) == (evidence == 'numeric_text')
    if evidence in ('none','numeric_text'): assert not observation['image'].any()
    if evidence == 'image_no_footer': assert (observation['image'][570:] == [243,246,252]).all()


def test_durable_reconnect_idempotency_and_conflict(tmp_path):
    s=spec(); b=bank(s); store=Store(tmp_path/'episodes.db', b)
    created=store.create(s['id']); assert created['ok']
    eid=created['episode_id']
    action={'type':'interact','event':{'kind':'set_transform','zoom':1.,'pan_x':160,'pan_y':0}}
    first=store.step(eid,'first',0,action)
    assert first['ok'] and first['result']['step']==1
    reopened=Store(tmp_path/'episodes.db',b)
    assert reopened.get(eid)['result']==first['result']
    assert reopened.step(eid,'first',0,action)==first
    with pytest.raises(ValueError): reopened.step(eid,'first',0,{'type':'observe'})
    with pytest.raises(ValueError): reopened.step(eid,'second',0,action)
    assert not {'checks','spec','rules','reward_ledger'} & first['result']['context'].keys()


def test_timeout_no_action_consumed(tmp_path,monkeypatch):
    s=spec(); store=Store(tmp_path/'episodes.db',bank(s))
    created=store.create(s['id']);eid=created['episode_id']
    monkeypatch.setattr(store,'worker',lambda *args: {'ok':False,'error':'evaluation_timeout','scored':False,'retryable':True})
    result=store.step(eid,'retryable',0,{'type':'observe'})
    assert not result['ok'] and not result['scored']
    assert store.get(eid)['result']['step']==0


def test_seal_tamper_rejected(tmp_path):
    b=bank(spec());b['records'][0]['spec']['mask']=1
    with pytest.raises(ValueError): Store(tmp_path/'episodes.db',b)


def test_wilson():
    lo,hi=wilson(0,30)
    assert lo==pytest.approx(0) and .1<hi<.12
    assert wilson(0,0)==[None,None]


def test_display_precision_claim_is_not_raw_grading():
    env=GeneratedRepairEnv(spec(),reward_mode='train');env.reset()
    from vision_sort.visual_training.tasks import canonical
    path='/projected/S1/0'
    ref=canonical(env._truth()['projected']['S1'][0]);cand=canonical(env.output['projected']['S1'][0])
    statement={'observation_id':env.frame_id,'source_revision':0,'path':path,
        'reference_json':json.dumps(ref),'candidate_json':json.dumps(cand),'verdict':'match' if ref==cand else 'mismatch'}
    row,reward=env._comparison(statement)
    assert row['correct'] and reward>=0


def test_rounded_prediction_matches_display():
    env=GeneratedRepairEnv(spec(),reward_mode='train');env.reset()
    from vision_sort.visual_training.tasks import canonical,transition
    event={'kind':'set_transform','zoom':1.,'pan_x':160,'pan_y':0}
    state=transition(env.family,env.state,event)
    expected=canonical(env._truth(state)['projected']['S1'][0])
    env.step({'type':'interact','event':event,'prediction':{'observation_id':env.frame_id,
        'path':'/projected/S1/0','value_json':json.dumps(expected)}})
    assert env.statement_ledger[-1]['correct']
    assert env.reward_ledger[-1]['components']['prediction']>=0


def test_real_worker_timeout_can_retry_same_decision(tmp_path):
    s=spec();store=Store(tmp_path/'episodes.db',bank(s));created=store.create(s['id'])
    eid=created['episode_id'];store.timeout=.001
    failed=store.step(eid,'same-decision',0,{'type':'observe'})
    assert not failed['ok'] and failed['error']=='evaluation_timeout'
    assert store.get(eid)['result']['step']==0
    store.timeout=45
    retried=store.step(eid,'same-decision',0,{'type':'observe'})
    assert retried['ok'] and retried['result']['step']==1


def test_no_dense_test_split_feedback(tmp_path):
    s=spec();s['split']='test';store=Store(tmp_path/'episodes.db',bank(s))
    with pytest.raises(ValueError):store.create(s['id'],mode='train')


def test_actor_recovers_saved_model_response_without_regeneration(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    actor=Path(__file__).resolve().parents[1]/'actor_client'
    monkeypatch.syspath_prepend(str(actor))
    loader=importlib.util.spec_from_file_location('v5_test_actor',actor/'api_agent.py')
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    public={'step':0,'context':{},'image_png_base64':'AA==','terminated':False,'truncated':False}
    class FakeClient:
        def __init__(self,*args):pass
        def create(self,*args):return {'episode_id':'episode','result':public}
        def new_request_id(self):return 'request'
        def step(self,*args):return {'ok':True,'result':dict(public,step=1,terminated=True)}
    class NoNewModelCall:
        def __init__(self,**kwargs):pass
        def call(self,*args):raise AssertionError('Saved successful model decision was regenerated')
    monkeypatch.setattr(module,'Client',FakeClient);monkeypatch.setattr(module,'DiscoveryCodexPolicy',NoNewModelCall)
    (tmp_path/'000-call.json').write_text(json.dumps({'status':'ok','completion_text':json.dumps({'action_json':'{"type":"submit"}','summary':'saved decision'})}))
    monkeypatch.setattr(sys,'argv',['api_agent.py','--task-id','test','--output',str(tmp_path)])
    module.main()
    state=json.loads((tmp_path/'checkpoint.json').read_text())
    assert state['pending'] is None and state['history'][0]['action']=={'type':'submit'}
