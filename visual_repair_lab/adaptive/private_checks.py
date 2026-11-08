"""Private evaluator gate; no code or configuration is exposed to the generator."""
import copy,itertools
from collections import defaultdict
from .certify import batch,decision_tree,visible_signature,adjacent
from .dsl import seal
from .private_cv import gold_source,initial_state,overlay,states,transition,valid,objects_equivalent as equivalent
from .detection_reference import detection_reference
import math


def validate_private(spec):
    required={'id','family','split','budget','scene_seed','pipeline'}
    if type(spec) is not dict or set(spec) not in (required,required|{'grading_seed'}):raise ValueError('Private fields')
    if 'grading_seed' in spec and (type(spec['grading_seed']) is not int or not 0<=spec['grading_seed']<2**64):raise ValueError('Private grading seed')
    if spec['family']!='detection_overlay' or spec['split']!='private_test':raise ValueError('Private authority only')
    if type(spec['scene_seed']) is not int or not 3510000<=spec['scene_seed']<3520000:raise ValueError('Private seed')
    if type(spec['budget']) is not int or not 12<=spec['budget']<=48:raise ValueError('Private budget')
    p=spec['pipeline']
    if set(p)!={'xywh','center','confidence','nms','swap','clip'}:raise ValueError('Private pipeline')
    for k in ('xywh','center','swap','clip'):
        if type(p[k]) is not bool:raise ValueError('Boolean required')
    if p['confidence'] not in (.45,.65) or p['nms'] not in (.35,.65):raise ValueError('Private literals')


def certify_private(spec):
    validate_private(spec);p=spec['pipeline'];cases=states(spec['scene_seed'],spec.get('grading_seed'))
    def truth(state,params=p):return detection_reference(state,params)
    outputs=[truth(s) for s in cases];runs=batch(gold_source(spec),cases)
    runtime=all(r['ok'] and valid(r['result']) and equivalent(r['result'],t) for r,t in zip(runs,outputs))
    choices=[dict(zip(('xywh','center','confidence','nms','swap','clip'),v)) for v in itertools.product((False,True),(False,True),(.45,.65),(.35,.65),(False,True),(False,True))]
    fingerprints=[seal([detection_reference(s,q) for s in cases]) for q in choices]
    unique={}
    for q,f in zip(choices,fingerprints):unique.setdefault(f,q)
    qs=list(unique.values());matrix=[[visible_signature(detection_reference(s,q)) for q in qs] for s in cases]
    costs=[0]+[1+len(s['events']) for s in cases[1:]]
    tree=decision_tree(matrix,costs)
    effects={}
    for k in p:
        mutant=dict(p)
        mutant[k]=not p[k] if type(p[k]) is bool else (.65 if p[k]==.45 else .45) if k=='confidence' else 1.
        effects[k]=sum(not equivalent(detection_reference(s,mutant),t) for s,t in zip(cases,outputs))
    stable=all(equivalent(t,adjacent(t,math.inf)) and equivalent(t,adjacent(t,-math.inf)) for t in outputs)
    # Some discrete flags can be inactive for malformed alternative encodings.
    # Released targets must witness every one of the six pipeline bugs.
    reasons=[]
    if not runtime:reasons.append('oracle_execution')
    if min(effects.values())<2:reasons.append('unwitnessed_private_rule')
    if tree['cost']+5>spec['budget']:reasons.append('ideal_budget_two_spare')
    if not stable:reasons.append('rounding_instability')
    return {'accepted':not reasons,'reasons':reasons,'spec_sha256':seal(spec),'oracle_execution':runtime,'independent_reference':runtime,
            'protected_cases':len(cases),'rule_effects':effects,'ideal_catalogue_classes':len(qs),
            'ideal_total_with_repair':tree['cost']+3,'spare_actions':spec['budget']-tree['cost']-3,
            'scope':'Held-out-family finite catalogue with ideal exact footer reader; no training feedback.',
            'behavior_fingerprint':seal(outputs)}
