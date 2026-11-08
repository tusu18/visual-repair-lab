"""V4 adversarial tests. Development fixtures only; no model calls or training."""
import copy
import json
import subprocess
import time
from pathlib import Path

import gymnasium
import pytest

from vision_sort.visual_causal.env import digest
from vision_sort.visual_training.env import VisualTrainingEnv, LIMITS, PROFILES
from vision_sort.visual_training.mutants import mutant_sources, compile_trusted, exposure
from vision_sort.visual_training.runtime import (
    run_program, checked_batch, EvaluationInfrastructureError, language_contract,
)
from vision_sort.visual_training.tasks import oracle_source, protected_states, expected, canonical, initial_state, transition

FAMILIES = ['drag_hysteresis','linked_brush','contact_dynamics']

def env(family='linked_brush', **kwargs):
    args=dict(family=family,split='development',candidate_mode='starter',reward_mode='train',reward_profile='full')
    args.update(kwargs)
    e=VisualTrainingEnv(**args)
    e.reset(seed=3110007,options={'world_seed':7})
    return e


def interact(e, **event):
    return e.step({'type':'interact','event':event})


def activate(e,source):
    e.step({'type':'edit','source':source});e.step({'type':'run'})


def claim(e,path,**extra):
    from vision_sort.visual_reward.env import field
    a,b=field(e._truth(),path),field(e.output,path)
    result=dict(observation_id=e.frame_id,source_revision=e.active_revision,path=path,
                reference_json=json.dumps(a),candidate_json=json.dumps(b),
                verdict='match' if a==b else 'mismatch')
    result.update(extra)
    return result

@pytest.mark.parametrize('family',FAMILIES)
def test_oracle_repair_and_per_case_status(family):
    e=env(family);activate(e,oracle_source(family,e._mask,e.split))
    e.step({'type':'submit'})
    assert e.final_result['success']
    assert all(r['execution_status']=='ok' for r in e.final_result['checks'])
    assert len(e._progress_status)==2

@pytest.mark.parametrize('family',FAMILIES)
def test_correct_candidate_needs_no_edit(family):
    e=env(family,candidate_mode='audit_correct')
    interact(e,kind='reset_scene')
    e.step({'type':'submit'})
    assert e.source_revision==0 and e.final_result['success']
    assert e.final_result['chosen_verification_tests']==1

@pytest.mark.parametrize('family',FAMILIES)
def test_incomplete_candidate_is_witnessed(family):
    e=env(family,candidate_mode='audit_incomplete')
    assert len(e._last_progress_passes)<len(e._progress_cases)

@pytest.mark.parametrize('family',FAMILIES)
def test_mutants_valid_and_banked_separately(family):
    sources,heldout=mutant_sources(family,511,'development')
    assert len(sources)==(6 if family=="drag_hysteresis" else 5) and len(heldout)==3 and not sources.keys() & heldout.keys()
    state=initial_state(family,3110007)
    for source in [*sources.values(),*heldout.values()]:
        result=run_program(source,state)
        assert result['ok'],result


def test_ordinary_zoom_does_not_kill_missing_clamp():
    e=env()
    # At zoom 1 the no-clamp program and truth are identical.
    interact(e,kind='set_transform',zoom=1,pan_x=0,pan_y=0)
    assert 'missing_zoom_clamp' not in e._metrics[-1]['structural_exposed']
    interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    assert 'missing_zoom_clamp' in e._metrics[-1]['structural_exposed']
    assert e.reward_ledger[-1]['components']['test_power']>0


def test_repeat_and_observe_cannot_farm_test_credit():
    e=env()
    event=dict(kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    interact(e,**event)
    before=e._positive.copy()
    e.step({'type':'observe'});interact(e,**event)
    assert e._positive==before


def test_comparison_no_extra_action_correct_false_and_stale():
    e=env();interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    c=claim(e,'/projected/S1/0/0'); n=e.steps
    e.step({'type':'observe','comparison':c})
    assert e.steps==n+1 and e.comparison_ledger[-1]['correct']
    assert e.reward_ledger[-1]['components']['comparison']>0
    e.step({'type':'observe','comparison':c})
    assert e.reward_ledger[-1]['components']['invalid']==-.1
    bad=claim(e,'/projected/S1/0/0',reference_json='123456')
    e.step({'type':'observe','comparison':bad})
    assert e.comparison_ledger[-1]['correct'] is False
    assert e.reward_ledger[-1]['components']['comparison']==-.1


def test_false_equality_is_penalized_and_exact_claim_does_not_mean_full_match():
    e=env();interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    c=claim(e,'/projected/S1/0/0',verdict='match')
    e.step({'type':'observe','comparison':c})
    assert not e.comparison_ledger[-1]['correct']


def test_actor_created_fault_cannot_earn_detection_bonus():
    e=env(candidate_mode='audit_correct')
    sources,_=mutant_sources(e.family,e._mask,e.split)
    activate(e,sources['missing_zoom_clamp'])
    interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    e.step({'type':'observe','comparison':claim(e,'/projected/S1/0/0')})
    assert e.comparison_ledger[-1]['correct']
    assert not e.comparison_ledger[-1]['original_defect']
    assert e.reward_ledger[-1]['components']['comparison']==0


def test_regression_and_restore_cannot_farm_progress():
    e=env();good=oracle_source(e.family,e._mask,e.split);bad=e.starter
    activate(e,good);first=e._positive['repair_progress']
    activate(e,bad)
    assert e.reward_ledger[-1]['components']['regression']<0
    activate(e,good)
    assert e._positive['repair_progress']==first
    assert e.reward_ledger[-1]['components']['repair_progress']==0


def test_new_edit_invalidates_verification():
    e=env();good=oracle_source(e.family,e._mask,e.split);activate(e,good)
    interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    assert e._structural_tests
    e.step({'type':'edit','source':good+'\n'})
    e.step({'type':'submit'})
    assert e.final_result['success']
    assert e.reward_ledger[-1]['components']['verification']==0

@pytest.mark.parametrize('profile',list(PROFILES))
def test_ablation_components_exact(profile):
    e=env(reward_profile=profile)
    interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    row=e.reward_ledger[-1]
    assert row['process_reward']==sum(row['components'].get(k,0) for k in PROFILES[profile])
    assert row['components']['information']==0


def test_positive_caps_do_not_allow_positive_failed_return():
    assert sum(LIMITS.values())==2.
    assert 2.-2.-.025<0
    e=env(max_steps=2);e.step({'type':'observe'});_,reward,t,tr,_=e.step({'type':'observe'})
    assert reward==-2.025 and tr and not t


def test_eval_and_train_same_actions_same_private_components():
    a=env(reward_mode='train'); b=env(reward_mode='evaluation')
    actions=[{'type':'interact','event':dict(kind='set_transform',zoom=.5,pan_x=0,pan_y=0)}, {'type':'observe'}]
    for action in actions:
        a.step(action);b.step(action)
        assert a.reward_ledger[-1]['components']==b.reward_ledger[-1]['components']
        assert b.reward==b.reward_ledger[-1]['base_reward']


def test_source_only_has_no_image_credit():
    e=env(image_access=False)
    interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    e.step({'type':'observe','comparison':claim(e,'/projected/S1/0/0')})
    assert e.reward_ledger[-1]['components']['comparison']==0
    assert e._positive['test_power']==0
    assert not e._observation()['image'].any()


def test_precise_contract_and_deterministic_fuel():
    assert 'remove' in language_contract()['methods']
    r=run_program('def solve(s):\n    return set()\n',{})
    assert r['error']=='Unsupported function: set.'
    src='def solve(s):\n    a=[1,2]\n    a.remove(1)\n    return {"a":a}\n'
    assert run_program(src,{})['result']=={'a':[2]}
    for _ in range(2):
        r=run_program('def solve(s):\n    while True:\n        pass\n',{})
        assert r['error_code']=='candidate_instruction_limit' and not r['infrastructure_error']


def test_batch_fresh_globals_and_same_semantics():
    src='items=[]\ndef solve(s):\n    items.append(s["x"])\n    return {"items":items}\n'
    states=[{'x':1},{'x':2}]
    results=checked_batch(src,states)
    assert [r['result'] for r in results]==[run_program(src,s)['result'] for s in states]


def test_worker_failure_abort_not_negative_training_example(monkeypatch):
    import vision_sort.visual_training.runtime as runtime
    e=env()
    def fail(*a,**k): raise subprocess.TimeoutExpired('worker',10)
    monkeypatch.setattr(runtime.subprocess,'run',fail)
    with pytest.raises(EvaluationInfrastructureError):
        interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    assert e._aborted and not e.reward_ledger
    with pytest.raises(gymnasium.error.ResetNeeded): e.step({'type':'observe'})


def test_batch_failure_no_partial_grade(monkeypatch):
    import vision_sort.visual_training.runtime as runtime
    e=env()
    def fail(*a,**k): raise OSError('private host path must never be emitted')
    monkeypatch.setattr(runtime.subprocess,'run',fail)
    with pytest.raises(EvaluationInfrastructureError) as exc: e.step({'type':'submit'})
    assert exc.value.code=='infrastructure_worker' and e.final_result is None
    assert not e.reward_ledger


def test_private_data_stays_out_of_observation():
    e=env()
    c=json.loads(e._observation()['context'])
    assert {'_mask','world_seed','mutants','progress_execution_status','candidate_kind','implementation_kind','reward_ledger'}.isdisjoint(c)
    assert c['language_contract'] and 'comparison' in c['action_examples']

@pytest.mark.parametrize('split,parity',[('train',0),('validation',1),('test',1)])
def test_composition_partition(split,parity):
    e=env(split=split)
    assert e._mask.bit_count()%2==parity
    assert e._implementation_kind in (('direct','renamed') if split=='train' else ('dispatch',))


def test_numeric_splits_disjoint():
    from vision_sort.visual_training.tasks import PARAMETERS
    for family in FAMILIES:
        for axis in (0,2) if family=='drag_hysteresis' else (0,1):
            assert not set(PARAMETERS['train'][family][axis]) & set(PARAMETERS['test'][family][axis])


def test_appearance_variation_preserves_information_and_scores():
    import numpy as np
    a=env();b=env(appearance='alternate')
    assert np.array_equal(a._image[:,:, [2,1,0]],b._image)
    assert a._truth()==b._truth()


def test_ocr_adapter_takes_only_rendered_pixels(monkeypatch):
    from visual_repair_lab import observations
    from visual_repair_lab.cli import _public_frame
    e=env()
    seen=[]
    def ocr(image):
        seen.append(image.copy());return 'text from actual pixels'
    monkeypatch.setattr(observations,'pixel_text',ocr)
    frame=_public_frame(e._observation(),observation_mode='ocr')
    assert frame['image'] is None and frame['pixel_derived_text']=='text from actual pixels'
    assert len(seen)==1 and not frame['context']['image_access']
    assert 'projected' not in frame['context']


def test_historical_failed_patches_have_stable_new_runtime_results():
    import hashlib
    fixtures=json.loads((Path(__file__).parent/'fixtures/historical-runtime.json').read_text())
    for item in fixtures:
        assert hashlib.sha256(item['source'].encode()).hexdigest()==item['source_sha256']
        first=checked_batch(item['source'],item['states'])
        second=checked_batch(item['source'],item['states'])
        assert first==second
        assert all(r['error_code']=='ok' for r in first)
        score=sum(canonical(r['result'])==canonical(t) for r,t in zip(first,item['truth']))
        assert score==item['expected_passed']
        if 'v2-linked' in item['name']:
            for _ in range(3):
                assert run_program(item['source'],item['states'][7])['result']==first[7]['result']


def test_oversized_results_are_candidate_errors_not_batch_crashes():
    source='def solve(s):\n    return {"value":"a"*10000}\n'
    assert run_program(source,{})['error_code']=='candidate_output'
    results=checked_batch(source,[{} for _ in range(100)])
    assert all(r['error_code']=='candidate_output' for r in results)


def test_negative_dimensions_fail_schema_without_renderer_exception():
    e=env('drag_hysteresis')
    s=oracle_source(e.family,e._mask,e.split)
    s=s.replace('90, 70','-90, 70')
    activate(e,s)
    assert not e.public_result['schema_valid']
    assert e.public_result['error']=='Output schema is invalid.'


def test_action_ledger_retains_attached_claims():
    e=env();interact(e,kind='set_transform',zoom=.5,pan_x=0,pan_y=0)
    c=claim(e,'/projected/S1/0/0')
    action={'type':'observe','comparison':c}
    e.step(action)
    assert e.research_report()['actions'][-1]==action


def test_cli_private_replay_record_is_separate_from_actor_stream(tmp_path,monkeypatch,capsys):
    import sys,io
    from visual_repair_lab.cli import main
    monkeypatch.setattr(sys,'stdin',io.StringIO('{"type":"observe"}\n'))
    target=tmp_path/'private-audit'
    assert main(['rollout','--split','development','--max-steps','1','--audit-dir',str(target)])==0
    public=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    private=json.loads((target/'private-reset.json').read_text())
    assert type(private['world_seed']) is int
    assert all('world_seed' not in row['context'] and 'report' not in row for row in public)
    assert (target/'0000.png').is_file() and (target/'0001.png').is_file()
    assert [json.loads((target/f'{i:04d}-public.json').read_text()) for i in range(2)] == public
    assert json.loads((target/'0001.json').read_text())['report']['actions']==[{'type':'observe'}]
