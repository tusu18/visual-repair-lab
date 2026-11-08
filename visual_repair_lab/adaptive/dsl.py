"""Typed, bounded train grammar. No model-authored executable code."""
import copy
import hashlib
import json

FAMILIES = ('drag_hysteresis', 'linked_brush', 'contact_dynamics')
BASE_MASKS = (0, 64, 128, 256)
HOOKS = {
    'drag_hysteresis': {'limit':'placement','gain':'pointer_move','threshold':'resize',
                        'snap':'pointer_up','offset':'draw_held','order':'pointer_move'},
    'linked_brush': {key:'projection' for key in ('limit','gain','threshold','snap','offset','order')},
    'contact_dynamics': {'limit':'velocity_update','gain':'gravity_update','threshold':'velocity_update',
                         'snap':'position_update','offset':'position_update','order':'integration_order'},
}
VALUES = {
    'drag_hysteresis': {'limit':[0.,10.], 'gain':[.5,.75], 'threshold':[460.,540.],
                        'snap':[10.,25.], 'offset':[7.,17.], 'order':[12.,24.]},
    'linked_brush': {'limit':[400.,500.], 'gain':[.8,1.2], 'threshold':[180.,260.],
                     'snap':[10.,20.], 'offset':[-15.,15.], 'order':[12.,24.]},
    'contact_dynamics': {'limit':[120.,240.], 'gain':[.5,1.5], 'threshold':[20.,80.],
                         'snap':[8.,16.], 'offset':[2.,5.], 'order':[True]},
}
CONDITIONS = {
    'drag_hysteresis': ('always','wide','compact','zoom_high','range'),
    'linked_brush': ('always','view_a','view_b','zoom_high','range'),
    'contact_dynamics': ('always','positive','negative','gravity_positive','range'),
}


def seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(spec):
    required={'id','family','world','rules','budget','starting_point','split','scene_seed'}
    if type(spec) is not dict or set(spec)!=required:
        raise ValueError('Invalid adaptive task fields')
    if not isinstance(spec['id'],str) or len(spec['id'])>96 or not spec['id'].replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid task ID')
    family=spec['family']
    if family not in FAMILIES or spec['split'] not in ('train','development'):
        raise ValueError('Generator cannot access private task families or splits')
    if type(spec['world']) is not dict or set(spec['world'])!={'mask'} or spec['world']['mask'] not in BASE_MASKS:
        raise ValueError('Unknown base world')
    if type(spec['budget']) is not int or not 8<=spec['budget']<=48:
        raise ValueError('Invalid action budget')
    if type(spec['scene_seed']) is not int or not 3410000<=spec['scene_seed']<3500000:
        raise ValueError('Use the adaptive train/development seed namespace')
    if spec['starting_point'] not in ('baseline','starter','proposal'):
        raise ValueError('Unknown starting point')
    if type(spec['rules']) is not list or not 1<=len(spec['rules'])<=3:
        raise ValueError('One to three rules required')
    seen=set()
    for rule in spec['rules']:
        if type(rule) is not dict or set(rule)!={'op','hook','condition','value'}:
            raise ValueError('Invalid rule fields')
        op=rule['op']
        if op not in HOOKS[family] or rule['hook']!=HOOKS[family][op] or op in seen:
            raise ValueError('Unknown/duplicate operation or invalid hook')
        if rule['condition'] not in CONDITIONS[family]:
            raise ValueError('Unknown condition')
        if not any(type(rule['value']) is type(v) and rule['value']==v for v in VALUES[family][op]):
            raise ValueError('Unsupported literal')
        seen.add(op)
    return copy.deepcopy(spec)
