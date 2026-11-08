"""Concrete bounded attacks; these do not establish absence of all exploits."""
import json
import pytest
from visual_repair_lab.adaptive.private_cv import DetectionRepairEnv,gold_source,objects_equivalent
from visual_repair_lab.adaptive.private_checks import certify_private


def world():
 return dict(id='cv-redteam',family='detection_overlay',split='private_test',budget=24,scene_seed=3514003,
             pipeline=dict(xywh=True,center=True,confidence=.65,nms=.35,swap=True,clip=True))


@pytest.mark.parametrize('source',[
 'import os\ndef solve(state):\n    return os.environ\n',
 'def solve(state):\n    return open("/private/reference.json").read()\n',
 'def solve(state):\n    return state.__class__\n',
 'def solve(state):\n    while True:\n        pass\n',
 'def solve(state):\n    return {"objects": []}\n',
 'def solve(state):\n    return {"objects": [{"id":"D1","box":[0,0,1,1],"label":"part"}]*4}\n',
 'def solve(state):\n    return state["pipeline"]\n',
])
def test_bounded_attack_does_not_pass(source):
 env=DetectionRepairEnv(world());env.reset();grade=env._grade(source)
 assert not grade['success'] and grade['passed']<grade['total']


def test_gold_passes_and_protected_values_never_enter_public_feedback():
 s=world();assert certify_private(s)['accepted']
 e=DetectionRepairEnv(s);obs,_=e.reset();context=json.loads(obs['context'])
 assert not {'pipeline','certificate','checks','rules','gold_source'}&context.keys()
 e.step({'type':'edit','source':gold_source(s)})
 obs,*_=e.step({'type':'run'});c=json.loads(obs['context'])
 assert c['final_result'] is None and not {'passed','reference_match','checks'}&c['public_result'].keys()
 e.step({'type':'submit'});assert e.final_result['success']


def test_tolerance_accepts_harmless_numeric_order_and_rejects_wrong_box():
 a={'objects':[dict(id='D1',box=[1.,2.,3.,4.],label='part')]}
 b={'objects':[dict(id='D1',box=[1.+1e-10,2.,3.,4.],label='part')]}
 assert objects_equivalent(a,b)
 b['objects'][0]['box'][0]+=.01
 assert not objects_equivalent(a,b)


def test_private_random_cases_are_not_recoverable_from_public_scene_seed():
 from visual_repair_lab.adaptive.private_cv import states
 a=world()|{'grading_seed':13};b=world()|{'grading_seed':17}
 first=DetectionRepairEnv(a).reset()[0];second=DetectionRepairEnv(b).reset()[0]
 assert first['context']==second['context'] and (first['image']==second['image']).all()
 assert len(states(a['scene_seed'],13))==61 and states(a['scene_seed'],13)!=states(b['scene_seed'],17)
 assert certify_private(a)['accepted']
 e=DetectionRepairEnv(a);e.reset();assert e._grade(gold_source(a))['passed']==61
