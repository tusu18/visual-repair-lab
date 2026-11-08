"""Bounded compositional generator and fail-closed release gate.

GPT supplies typed proposals, not code. Only reviewed template operations execute.
This expands rule compositions and parameters, not the number of task families.
"""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

from vision_sort.visual_training import tasks
from vision_sort.visual_training.mutants import compile_trusted
from vision_sort.visual_training.runtime import checked_batch
from . import templates
from .compare import equivalent

RULES = {
    'drag_hysteresis': {'canvas_clamp': [True, False], 'wide_gain': [.35, .5, .65, .8, 1.]},
    'linked_brush': {'pan_limit': [12., 20., 28., 36.], 'column_a': [True, False]},
    'contact_dynamics': {'dt_max': [.025, .035, .04, .045], 'gravity_limit': [350., 450., 600., 750.]},
}
LEVELS = {'foundation': 48, 'composed': 32, 'hard': 16}
SPLITS = ('train', 'validation', 'test')
NEUTRAL = dict(canvas_clamp=False, wide_gain=1., pan_limit=1000000., column_a=False, dt_max=1., gravity_limit=1000000.)


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_spec(spec):
    if type(spec) is not dict or set(spec) != {'id', 'family', 'split', 'level', 'mask', 'scene_seed', 'rules'}:
        raise ValueError('Invalid proposal fields')
    if not isinstance(spec['id'], str) or not spec['id'].replace('-', '').replace('_', '').isalnum() or len(spec['id']) > 80:
        raise ValueError('Invalid proposal ID')
    if spec['family'] not in RULES or spec['split'] not in SPLITS or spec['level'] not in LEVELS:
        raise ValueError('Unknown family, split or level')
    if type(spec['mask']) is not int or not 0 <= spec['mask'] < 512:
        raise ValueError('Invalid mask')
    if type(spec['scene_seed']) is not int or not 3310000 <= spec['scene_seed'] < 3400000:
        raise ValueError('Use the v5 reserved scene namespace')
    if spec['level'] == 'foundation' and spec['mask'] & 63:
        raise ValueError('Foundation fixes older factors')
    rule_set = RULES[spec['family']]
    if type(spec['rules']) is not dict or set(spec['rules']) != set(rule_set):
        raise ValueError('Invalid rule fields')
    for key, value in spec['rules'].items():
        if not any(type(value) is type(v) and value == v for v in rule_set[key]):
            raise ValueError('Unsupported rule value')
    return copy.deepcopy(spec)


def source_for(spec, disabled=()):
    """Compile exclusively reviewed source templates with validated literals."""
    validate_spec(spec)
    family = spec['family']
    constants = dict(NEUTRAL, **spec['rules'])
    for key in disabled:
        constants[key] = NEUTRAL[key]
    fns = {'drag_hysteresis': [templates._place, templates._drag],
           'linked_brush': [templates._projection, tasks._segment_hit, templates._chart],
           'contact_dynamics': [templates._physics]}[family]
    body = '\n\n'.join(inspect.getsource(f) for f in fns)
    for key, value in constants.items():
        body = body.replace(key.upper(), repr(value))
    bits = [(spec['mask'] >> i) & 1 for i in range(6)]
    args = tasks.parameters(family, spec['mask'], spec['split'])
    return body + f'\n\ndef solve(state):\n    return {fns[-1].__name__}(state, {bits!r}, ' + ', '.join(repr(x) for x in args) + ')\n'


def sources(spec):
    return source_for(spec), {key: source_for(spec, [key]) for key in spec['rules']
                             if spec['rules'][key] != NEUTRAL[key]}


def states_for(spec, pool):
    family = spec['family']
    states = tasks.protected_states(family, spec['scene_seed'], spec['split'], pool=pool)
    # Boundary cases are compiled with the rules, so new behavior cannot silently
    # enter the release without corresponding grading and progress witnesses.
    sequences = []
    if family == 'drag_hysteresis':
        widths = [280, 285] if pool == 'final' else [281, 286]
        sequences += [[dict(kind='resize', width=w)] for w in widths]
        for dx in ([24, 44] if pool == 'final' else [26, 46]):
            sequences.append([dict(kind='resize', width=640), dict(kind='down', x=60, y=290),
                              dict(kind='move', x=60+dx, y=290)])
    elif family == 'linked_brush':
        for pan in ([-160, 160] if pool == 'final' else [-140, 140]):
            sequences.append([dict(kind='set_transform', zoom=1., pan_x=pan, pan_y=-pan)])
        for x in ([0, 30] if pool == 'final' else [10, 40]):
            sequences.append([dict(kind='set_view', context='A'),
                dict(kind='brush', x0=x, x1=580, y0=100, y1=101, mode='replace')])
    else:
        for gravity in ([-1000, 1000] if pool == 'final' else [-950, 950]):
            sequences.append([dict(kind='set_physics', gravity=gravity, dt=.02), dict(kind='advance', ticks=2)])
        for speed in ([100, 180] if pool == 'final' else [110, 190]):
            sequences.append([dict(kind='set_body', x=200, y=180, vx=speed, vy=speed),
                dict(kind='set_physics', gravity=0, dt=.05), dict(kind='advance', ticks=2)])
    for events in sequences:
        state = tasks.initial_state(family, spec['scene_seed'])
        for event in events:
            state = tasks.transition(family, state, event)
        states.append(state)
    return states


def certify(spec, minimum_kills=2):
    """Reject unwitnessed rules. No claim of globally minimal experiment count.

    Witnesses establish distinguishability from omission mutants within the
    published action budget. Oracle execution establishes implementability.
    Neither establishes that a nonprivileged model will discover that program.
    """
    validate_spec(spec)
    gold, bugs = sources(spec)
    solve = compile_trusted(gold)
    final = states_for(spec, 'final')
    progress = states_for(spec, 'progress')
    neutral = compile_trusted(source_for(spec, spec['rules']))
    base = compile_trusted(tasks.oracle_source(spec['family'], spec['mask'], spec['split']))
    parity = all(equivalent(neutral(copy.deepcopy(s)), base(copy.deepcopy(s))) for s in final + progress)
    results = checked_batch(gold, final)
    oracle = all(r['ok'] and tasks.valid(spec['family'], s, r['result']) and equivalent(r['result'], solve(copy.deepcopy(s)))
                 for s, r in zip(final, results))
    witnesses = {}
    reasons = []
    if not parity:
        reasons.append('neutral_reference_parity')
    if not oracle:
        reasons.append('oracle_execution')
    for name, source in bugs.items():
        bug = compile_trusted(source)
        hits = [i for i, s in enumerate(final) if not equivalent(solve(copy.deepcopy(s)), bug(copy.deepcopy(s)))]
        available = [i for i in hits if len(final[i]['events']) + 3 <= LEVELS[spec['level']]]
        # A visually distinct rendering is required, not merely a numeric mismatch.
        visible = [i for i in available if (tasks.panel(spec['family'], final[i], solve(copy.deepcopy(final[i])), 'A') !=
                    tasks.panel(spec['family'], final[i], bug(copy.deepcopy(final[i])), 'A')).any()]
        witnesses[name] = {'killed_checks': len(hits), 'visible_within_budget': visible,
                           'shortest_witness_actions_plus_edit_run_submit': min((len(final[i]['events']) + 3 for i in visible), default=None)}
        if len(hits) < minimum_kills or not visible:
            reasons.append('unwitnessed_rule:' + name)
    # A concrete joint witness schedule must fit, including reset costs and
    # edit/run/submit. This certifies the finite added-rule omission challenge.
    selected, remaining = [], set(witnesses)
    while remaining:
        options = []
        for i, state in enumerate(final):
            covered = {name for name in remaining if i in witnesses[name]['visible_within_budget']}
            if covered:
                cost = len(state['events']) + 1  # reset before each experiment
                options.append((len(covered)/cost, -cost, -i, covered))
        if not options:
            break
        _, _, negative_i, covered = max(options, key=lambda v: v[:3])
        selected.append(-negative_i)
        remaining -= covered
    witness_cost = 3 + sum(len(final[i]['events'])+1 for i in selected)
    if remaining or witness_cost > LEVELS[spec['level']]:
        reasons.append('joint_witness_budget')
    return {'accepted': not reasons, 'reasons': reasons, 'spec_sha256': seal(spec),
            'oracle_sha256': hashlib.sha256(gold.encode()).hexdigest(), 'oracle_checks': len(final),
            'oracle_pass': oracle, 'neutral_parity': parity, 'rule_witnesses': witnesses,
            'joint_witness_cases': selected, 'joint_witness_actions_including_repair': witness_cost,
            'scope': 'finite protected bank; omission-mutant distinguishability, not full identifiability or autonomous solvability'}


def release_bank(proposals, path):
    if not proposals or len({p['id'] for p in proposals}) != len(proposals):
        raise ValueError('Empty or duplicate bank')
    # A complete exact configuration may belong to only one split.
    seen = {}
    for p in proposals:
        key = seal({k: p[k] for k in ('family', 'mask', 'rules')})
        if key in seen and seen[key] != p['split']:
            raise ValueError('Cross-split configuration overlap')
        seen[key] = p['split']
    records = [{'spec': p, 'certificate': certify(p)} for p in proposals]
    accepted = [r for r in records if r['certificate']['accepted']]
    payload = {'version': '5.0.0', 'records': accepted, 'rejected': [r for r in records if not r['certificate']['accepted']]}
    payload['bank_sha256'] = seal(accepted)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p, d = successes / total, 1 + z*z/total
    c = (p + z*z/(2*total))/d
    r = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0., c-r), min(1., c+r)]
