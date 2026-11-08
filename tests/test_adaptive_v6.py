import copy,json,random
import pytest
from visual_repair_lab.adaptive.dsl import HOOKS,VALUES,seal,validate
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.adaptive.certify import certify,palette,source_allow_empty,visible_signature
from visual_repair_lab.adaptive.curriculum import Curriculum,summarize_group,signal_summary
from visual_repair_lab.adaptive.generator import edit_task,filter_records
from visual_repair_lab.adaptive.env import AdaptiveRepairEnv
from visual_repair_lab.adaptive.probes import experiments,state_for,grading_states
from visual_repair_lab.adaptive.pool_gate import certify_against_catalogue

def pool_certificate(s):
    c=certify(s);r={'spec':s,'certificate':c};c['whole_catalogue_gate']=certify_against_catalogue(r,[r]);return c
from vision_sort.visual_training.mutants import compile_trusted


def spec(family='drag_hysteresis',op='offset',budget=32):
    return dict(id=family+'-'+op,family=family,world={'mask':0},rules=[dict(op=op,hook=HOOKS[family][op],condition='always',value=VALUES[family][op][0])],
                budget=budget,starting_point='baseline',split='train',scene_seed=3410001)


@pytest.mark.parametrize('family',['drag_hysteresis','linked_brush','contact_dynamics'])
def test_six_blocks_compile_and_disable_to_original(family):
    from vision_sort.visual_training.tasks import oracle_source
    for op in HOOKS[family]:
        s=spec(family,op);compile_trusted(compile_source(s))
        assert compile_source(s,disabled=[0])==oracle_source(family,0,'train')


@pytest.mark.parametrize('family',['drag_hysteresis','linked_brush','contact_dynamics'])
def test_ideal_policy_observations_identify_passing_source(family):
    s=spec(family);c=certify(s);assert c['accepted'],c['reasons']
    target=compile_trusted(compile_source(s));tree=c['private_tree']
    sequences=[[]]+[[{'kind':'reset_scene'}]+seq for seq in experiments(family)]
    actions=0
    while 'leaf' not in tree:
        seq=sequences[tree['experiment']];actions+=len(seq)
        observation=target(state_for(family,s['scene_seed'],seq))
        tree=tree['branches'][visible_signature(observation)]
    candidate=palette(s)[c['private_representatives'][tree['leaf']]]
    program=source_allow_empty(candidate)
    env=AdaptiveRepairEnv(s);env.reset();env.step({'type':'edit','source':program});env.step({'type':'run'});env.step({'type':'submit'})
    assert env.final_result['success'] and actions+3+2<=s['budget']


def attempts(successes,interrupted=False):
    return [dict(attempt_id=str(i),status='interrupted' if interrupted and i==7 else 'completed',success=i<successes,**{'return':float(i)}) for i in range(8)]


@pytest.mark.parametrize('wins,state,keep',[(0,'parked',False),(4,'active',True),(7,'active',True),(8,'retired',False)])
def test_group_estimates(wins,state,keep):
    s=summarize_group(attempts(wins));assert s['p']==wins/8 and s['state']==state
    assert s['binary_training_keep']==keep and s['priority']==s['p']*(1-s['p'])


def test_interruptions_are_not_model_failures():
    assert not summarize_group(attempts(4,True))['eligible']


def test_duplicate_attempts_rejected():
    rows=attempts(4);rows[-1]=rows[0]
    with pytest.raises(ValueError):summarize_group(rows)


def test_signal_counts_before_filtering():
    rows=[{'summary':summarize_group(attempts(k))} for k in (0,4,8)]
    s=signal_summary(rows);assert s['raw_mixed_share']==pytest.approx(1/3) and s['kept_mixed_share']==1.


def test_policy_scoped_persistence_and_rng(tmp_path):
    s=spec();record={'spec':s,'certificate':pool_certificate(s)}
    c=Curriculum(tmp_path/'c.db','policy-v1');c.add(record);c.record(s['id'],0,attempts(4))
    assert c.choose()[1]['reason']=='p_times_one_minus_p'
    restored=Curriculum(tmp_path/'c.db','policy-v1');assert restored.snapshot()[s['id']]['estimate']['p']==.5
    new_policy=Curriculum(tmp_path/'c.db','policy-v2');assert 'estimate' not in new_policy.snapshot()[s['id']]


def test_private_tasks_cannot_enter_train_generator_or_picker(tmp_path):
    s=spec();s['split']='private_test'
    with pytest.raises(ValueError):validate(s)
    c=Curriculum(tmp_path/'c.db','p')
    with pytest.raises(ValueError):c.add({'spec':s,'certificate':{'accepted':True,'spec_sha256':seal(s)}})


def test_parked_not_sampled_as_active(tmp_path):
    s=spec();c=Curriculum(tmp_path/'c.db','p');c.add({'spec':s,'certificate':pool_certificate(s)});c.record(s['id'],0,attempts(0))
    assert c.choose()[0] is None


def test_editor_moves_difficulty_and_requires_recertification():
    s=spec();easy,lineage=edit_task(s,0.,random.Random(1),1)
    assert easy['budget']>s['budget'] and lineage['operation']=='add_budget'
    hard,lineage=edit_task(s,.875,random.Random(1),2)
    assert len(hard['rules'])>len(s['rules']) or hard['budget']<s['budget']


def test_no_effect_rules_rejected():
    s=spec('contact_dynamics','order');s['world']['mask']=256
    c=certify(s);assert not c['accepted'] and any(r.startswith('rule_effect') for r in c['reasons'])


def test_duplicate_behavior_rejected():
    s=spec();c=certify(s)
    rows=[{'spec':s,'certificate':copy.deepcopy(c)},{'spec':dict(s,id='another'),'certificate':copy.deepcopy(c)}]
    r=filter_records(rows);assert len(r['records'])==1 and r['rejection_counts']['duplicate_behavior']==1


def test_adaptive_progress_and_post_edit_power():
    s=spec();e=AdaptiveRepairEnv(s,reward_mode='train');e.reset()
    e.step({'type':'interact','event':{'kind':'down','x':80,'y':250}})
    assert e.reward_ledger[-1]['components']['test_power']==0
    e.step({'type':'edit','source':compile_source(s)})
    assert e.reward_ledger[-1]['components']['repair_progress']>0
    e.step({'type':'run'});e.step({'type':'interact','event':{'kind':'down','x':80,'y':250}})
    assert e.reward_ledger[-1]['components']['test_power']>0


def test_private_cv_runtime_and_hidden_contract():
    from visual_repair_lab.adaptive.private_checks import certify_private
    from visual_repair_lab.adaptive.private_cv import DetectionRepairEnv,gold_source
    s=dict(id='cv-dev',family='detection_overlay',split='private_test',budget=24,scene_seed=3510000,
           pipeline=dict(xywh=True,center=True,confidence=.45,nms=.35,swap=False,clip=True))
    c=certify_private(s);assert c['accepted'] and min(c['rule_effects'].values())>=2
    e=DetectionRepairEnv(s);obs,_=e.reset();context=json.loads(obs['context']);assert e.observation_space.contains(obs)
    assert not {'pipeline','world','rules','certificate'} & context.keys()
    e.step({'type':'edit','source':gold_source(s)});e.step({'type':'run'});e.step({'type':'submit'})
    assert e.final_result['success']


def test_whole_catalogue_gate_has_no_per_task_operation_hint():
    a=spec('drag_hysteresis','offset');b=spec('drag_hysteresis','threshold')
    records=[{'spec':s,'certificate':certify(s)} for s in (a,b)]
    c=certify_against_catalogue(records[0],records)
    assert c['accepted'] and c['operation_palette_hint'] is False and c['catalogue_programs']==2


def test_picker_rejects_only_local_palette_certificate(tmp_path):
    s=spec();c=Curriculum(tmp_path/'c.db','p')
    with pytest.raises(ValueError,match='whole-catalogue'):
        c.add({'spec':s,'certificate':certify(s)})


def test_pool_gate_can_distinguish_history_conditioned_thresholds():
    a=spec('drag_hysteresis','threshold');a['rules'][0]['value']=540.
    b=copy.deepcopy(a);b['id']='compact-only';b['rules'][0]['condition']='compact'
    records=[{'spec':s,'certificate':certify(s)} for s in (a,b)]
    c=certify_against_catalogue(records[0],records)
    assert c['accepted'] and c['ideal_total_with_repair']+2<=a['budget']
    assert any(len(s)==3 and s[1].get('width')==640 for s in c['private_sequences'])


def test_persistent_teacher_uses_external_eight_attempt_outcomes(tmp_path):
    from visual_repair_lab.adaptive.teacher import AdaptiveTeacher
    s=spec();t=AdaptiveTeacher(tmp_path/'teacher.db','checkpoint-1')
    t.curriculum.add({'spec':s,'certificate':pool_certificate(s)})
    r=t.observe_group(s['id'],1,attempts(4));assert r['mixed'] and r['p']==.5
    restored=AdaptiveTeacher(tmp_path/'teacher.db','checkpoint-1')
    chosen,why=restored.next_task();assert chosen['spec']['id']==s['id'] and why['reason']=='p_times_one_minus_p'


def test_teacher_attempt_generation_is_bounded_and_auditable(tmp_path,monkeypatch):
    from visual_repair_lab.adaptive.teacher import AdaptiveTeacher
    t=AdaptiveTeacher(tmp_path/'teacher.db','checkpoint-2')
    monkeypatch.setattr(t,'_accept',lambda spec,lineage:{'certificate':{'accepted':False}})
    record,why=t.next_task(max_generation_attempts=2)
    assert record is None and why['attempts']==2


def test_public_factory_and_gym_seed_preserve_fixed_task():
    import gymnasium as gym
    import visual_repair_lab
    s=spec();env=visual_repair_lab.make_env(spec=s)
    a,_=env.reset(seed=7);assert env.np_random_seed==7
    b,_=env.reset(seed=17);assert env.np_random_seed==17
    assert (a['image']==b['image']).all()
    assert 'VisualAdaptiveRepair-v0' in gym.envs.registry


def test_mastered_pool_requires_hardening_not_constant_replay(tmp_path):
    s=spec();c=Curriculum(tmp_path/'c.db','p',replay_probability=0.)
    c.add({'spec':s,'certificate':pool_certificate(s)});c.record(s['id'],0,attempts(8))
    chosen,why=c.choose();assert chosen is None and why['reason']=='all_mastered_generate_or_harden'


def test_packaged_bank_can_start_official_evaluator(tmp_path):
    from pathlib import Path
    from visual_repair_lab.adaptive.server import Store
    bank=json.loads((Path(__file__).resolve().parents[1]/'banks/train.json').read_text())
    store=Store(tmp_path/'episodes.db',bank)
    assert len(store.bank)==134


def test_private_footer_free_evidence_preserves_public_context():
    import numpy as np
    from visual_repair_lab.adaptive.private_cv import DetectionRepairEnv
    s=dict(id='cv-dev',family='detection_overlay',split='private_test',budget=24,scene_seed=3510000,
           pipeline=dict(xywh=True,center=True,confidence=.45,nms=.35,swap=False,clip=True))
    full,_=DetectionRepairEnv(s,evidence='image').reset()
    visual,_=DetectionRepairEnv(s,evidence='image_no_footer').reset()
    assert full['context']==visual['context'] and np.array_equal(full['image'][:550],visual['image'][:550])
    assert (visual['image'][550:]==[243,246,252]).all() and not np.array_equal(full['image'][550:],visual['image'][550:])


def test_detection_grading_is_order_independent_but_identity_preserving():
    from visual_repair_lab.adaptive.private_cv import objects_equivalent,valid
    rows=[dict(id='D1',box=[0.,1.,2.,3.],label='part'),dict(id='D2',box=[4.,5.,6.,7.],label='defect')]
    assert objects_equivalent({'objects':rows},{'objects':rows[::-1]})
    assert not valid({'objects':[rows[0],rows[0]]})
    changed=copy.deepcopy(rows);changed[0]['box'][0]+=1.
    assert not objects_equivalent({'objects':rows},{'objects':changed})


def test_comment_only_edit_cannot_farm_test_power_or_verification():
    e=AdaptiveRepairEnv(spec(),reward_mode='train');e.reset()
    e.step({'type':'edit','source':e.source+'\n# no behavior change\n'});e.step({'type':'run'})
    e.step({'type':'interact','event':{'kind':'down','x':80,'y':250}})
    assert e.reward_ledger[-1]['components']['test_power']==0 and not e._final_tests


def test_private_scene_seeds_vary_evidence_and_reset_reproduces_it():
    from visual_repair_lab.adaptive.private_cv import initial_state,transition,states
    a,b=initial_state(3514001),initial_state(3514002)
    assert a['detections']!=b['detections']
    reset=transition(a,{'kind':'reset_scene'});reset['events']=[]
    assert reset==a and states(3514001)!=states(3514002)


def test_independent_reference_matches_all_primitives_on_public_experiments():
    from visual_repair_lab.adaptive.reference import reference
    from visual_repair_lab.v5.compare import equivalent
    for family in HOOKS:
        for op in HOOKS[family]:
            s=spec(family,op);fn=compile_trusted(compile_source(s))
            for state in grading_states(s,'final'):
                assert equivalent(reference(s,state),fn(copy.deepcopy(state)))


def test_independent_detection_reference_matches_executable_repair():
    from visual_repair_lab.adaptive.private_cv import overlay,states,objects_equivalent
    from visual_repair_lab.adaptive.detection_reference import detection_reference
    settings=dict(xywh=True,center=True,confidence=.65,nms=.35,swap=True,clip=True)
    for state in states(3514001):assert objects_equivalent(overlay(copy.deepcopy(state),**settings),detection_reference(state,settings))


def test_conditional_order_does_not_persist_after_guard_or_reset():
    from visual_repair_lab.v5.compare import equivalent
    s=spec('contact_dynamics','order');s['world']['mask']=64
    s['rules'][0].update(condition='range',value=True)
    fn=compile_trusted(compile_source(s))
    first=[{'kind':'set_body','x':203,'y':150,'vx':310,'vy':-500},{'kind':'set_physics','gravity':800,'dt':.025},{'kind':'advance','ticks':2}]
    tail=[{'kind':'set_body','x':203,'y':150,'vx':310,'vy':700},{'kind':'set_physics','gravity':800,'dt':.025},{'kind':'advance','ticks':2}]
    assert equivalent(fn(state_for(s['family'],s['scene_seed'],first+[{'kind':'reset_scene'}]+tail)),fn(state_for(s['family'],s['scene_seed'],[{'kind':'reset_scene'}]+tail)))
    assert certify(s)['reset_history_invariance']


def test_curriculum_rejects_environment_changes_without_overwriting_history(tmp_path):
    path=tmp_path/'curriculum.db';c=Curriculum(path,'p')
    with c.connect() as db:db.execute('UPDATE environment_metadata SET seal=?',('different-runtime',))
    with pytest.raises(ValueError,match='Environment changed'):Curriculum(path,'p')


def test_ideal_certificate_charges_the_observed_branch_and_executes_repair():
    records=[]
    for condition in ('always','view_a','view_b'):
        s=spec('linked_brush','offset');s['id']='chart-'+condition;s['rules'][0]['condition']=condition
        records.append({'spec':s,'certificate':certify(s)})
    c=certify_against_catalogue(records[0],records)
    assert c['accepted'] and c['executed_ideal_valid']
    assert c['ideal_total_with_repair']<c['ideal_worst_total_with_repair']
