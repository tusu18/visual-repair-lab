"""Private variable-mechanism truth and disjoint development grading pools.

Only CONTRACTS, CONTROLS, starter code and public transition state are actor
inputs. Oracle functions, factor names, parameter choices and test pools are
trusted evaluator data. The parameter split is separate from the scene seed.
"""
import copy
import inspect
import random

from ..visual_batch4.contracts import CONTRACTS as OLD_CONTRACTS
from ..visual_causal.tasks import (
    CONTROLS as CONTROLS,
)
from ..visual_causal.tasks import FACTORS as ORIGINAL_FACTORS
from ..visual_causal.tasks import (
    FAMILIES as FAMILIES,
)
from ..visual_causal.tasks import (
    canonical as canonical,
)
from ..visual_causal.tasks import (
    initial_state,
    transition,
)
from ..visual_causal.tasks import (
    panel as panel,
)
from ..visual_causal.tasks import (
    starter_source as starter_source,
)
from ..visual_causal.tasks import (
    valid as valid,
)

FACTORS = {
    'drag_hysteresis': ORIGINAL_FACTORS['drag_hysteresis'] + (
        'held_lift', 'zoom_scaled_lift', 'snap_grid'),
    'linked_brush': ORIGINAL_FACTORS['linked_brush'] + (
        'lower_zoom_bound', 'upper_zoom_bound', 'zoom_scaled_pan'),
    'contact_dynamics': ORIGINAL_FACTORS['contact_dynamics'] + (
        'negative_speed_cap', 'positive_speed_cap', 'cap_before_gravity'),
}

PARAMETERS = {
    'train': {
        'drag_hysteresis': ((5, 13), (False, True), (11, 29)),
        'linked_brush': ((.57, .79), (1.27, 1.63), (False, True)),
        'contact_dynamics': ((350, 550), (430, 670), (False, True)),
    },
    'validation': {
        'drag_hysteresis': ((8, 15), (False, True), (17, 33)),
        'linked_brush': ((.63, .87), (1.37, 1.73), (False, True)),
        'contact_dynamics': ((410, 590), (470, 710), (False, True)),
    },
    'test': {
        'drag_hysteresis': ((9, 18), (False, True), (14, 27)),
        'linked_brush': ((.61, .83), (1.47, 1.87), (False, True)),
        'contact_dynamics': ((450, 630), (510, 750), (False, True)),
    },
}
# Development uses training parameter ranges; validation and test numerical
# choices are distinct. None of these worlds constitute an evaluation result.
PARAMETERS['development'] = PARAMETERS['train']

CONTRACTS = dict(OLD_CONTRACTS)
CONTRACTS['drag_hysteresis'] = CONTRACTS['drag_hysteresis'].replace(
    'Set_zoom never changes how cards are drawn. ',
    'Set_zoom changes the public zoom control. ').replace(
    'Down hit-tests current cards in reversed ID order and holds the hit card.',
    'Down attempts to hold a card, resolving overlapping hits in reversed ID order.'
).replace(
    'Move events while a card is held update its offset',
    'Move events while a card is held may update its offset')
CONTRACTS['linked_brush'] = CONTRACTS['linked_brush'].replace(
    'the overview projection is x=70+460*zoom*xvalue+pan_x, '
    'y=200+300*zoom*(1-yvalue)+pan_y. ',
    'the view transform affects the projected coordinates. ')
CONTRACTS = {key: value + ' All scored fields are visibly drawn. Choose your own '
             'experiments and tests, including interactions after code changes.'
             for key, value in CONTRACTS.items()}


def parameters(family, mask, split='train'):
    if family not in FAMILIES or split not in PARAMETERS:
        raise ValueError('Unknown task family or parameter split')
    if type(mask) is not int or not 0 <= mask < 512:
        raise ValueError('World mask must be an integer in [0, 512)')
    return tuple(choices[(mask >> (6 + i)) & 1]
                 for i, choices in enumerate(PARAMETERS[split][family]))


def _drag(s, bits, lift, lift_scales, snap_grid):
    width, zoom, mode = 640, 1.0, 'compact'
    offsets = {key: [0.0, 0.0] for key in s['ids']}
    active, origin, start = None, [0.0, 0.0], [0.0, 0.0]
    positions = {}
    for e in [{'kind': 'initial'}] + s['events']:
        kind = e['kind']
        if kind == 'reset_scene':
            width, zoom, mode = 640, 1.0, 'compact'
            offsets = {key: [0.0, 0.0] for key in s['ids']}
            active = None
        if kind == 'resize':
            width = e['width']
            high = [480, 520][bits[0]]
            low = high - [0, 40][bits[1]]
            if mode == 'compact' and width > high:
                mode = 'wide'
            elif mode == 'wide' and width < low:
                mode = 'compact'
        if kind == 'set_zoom':
            zoom = e['zoom']
        positions = {}
        for i, key in enumerate(s['ids']):
            bx = 60 + (i % 2) * 140 if mode == 'compact' else 40 + i * 130
            by = 220 + (i // 2) * 110 if mode == 'compact' else 270
            positions[key] = [bx + offsets[key][0], by + offsets[key][1], 90, 70]
        if kind == 'down':
            active = None
            for key in reversed(s['ids']):
                x, y, w, h = positions[key]
                if x <= e['x'] <= x + w and y <= e['y'] <= y + h:
                    active = key
                    origin = [e['x'], e['y']]
                    start = offsets[key][:]
                    break
        if kind == 'move' and active is not None:
            x, y, w, h = positions[active]
            if not bits[2] and not (x <= e['x'] <= x + w and y <= e['y'] <= y + h):
                active = None
            else:
                divisor = zoom if bits[5] else 1
                offsets[active] = [start[0] + (e['x'] - origin[0]) / divisor,
                                   start[1] + (0 if bits[4] else (e['y'] - origin[1]) / divisor)]
        if kind == 'up' and active is not None:
            if bits[3]:
                offsets[active] = [round(v / snap_grid) * snap_grid for v in offsets[active]]
            active = None
        # Return the changed positions in this frame, not one event later.
        for i, key in enumerate(s['ids']):
            bx = 60 + (i % 2) * 140 if mode == 'compact' else 40 + i * 130
            by = 220 + (i // 2) * 110 if mode == 'compact' else 270
            positions[key] = [bx + offsets[key][0], by + offsets[key][1], 90, 70]
    if active is not None:
        positions[active][1] -= lift * (zoom if lift_scales else 1)
    return {'positions': positions, 'active_id': active, 'layout': mode}


def _projection(frame, bits, pan_scales):
    context, zoom = frame['context'], frame['zoom']
    power, reverse = 1, False
    if context != 'overview':
        off = 0 if context == 'A' else 2
        power, reverse = [1, 2][bits[off]], bool(bits[off + 1])
    out = {}
    for key, points in frame['series'].items():
        out[key] = [[70 + 460 * zoom * x ** power + frame['pan_x'] * (zoom if pan_scales else 1),
                     200 + 300 * zoom * (y if reverse else 1 - y) + frame['pan_y'] * (zoom if pan_scales else 1)]
                    for x, y in points]
    return out


def _segment_hit(a, b, box):
    lo, hi = 0.0, 1.0
    for p, q in [(-(b[0]-a[0]), a[0]-box[0]), (b[0]-a[0], box[2]-a[0]),
                 (-(b[1]-a[1]), a[1]-box[1]), (b[1]-a[1], box[3]-a[1])]:
        if p == 0:
            if q < 0:
                return False
        elif p < 0:
            lo = max(lo, q/p)
        else:
            hi = min(hi, q/p)
        if lo > hi:
            return False
    return True


def _chart(s, bits, min_zoom, max_zoom, pan_scales):
    frame = {'context': 'overview', 'zoom': 1.0, 'pan_x': 0.0, 'pan_y': 0.0, 'series': s['series']}
    brushes = []
    for e in s['events']:
        if e['kind'] == 'reset_scene':
            frame = {'context': 'overview', 'zoom': 1.0, 'pan_x': 0.0, 'pan_y': 0.0, 'series': s['series']}
            brushes = []
        elif e['kind'] == 'set_view':
            frame['context'] = e['context']
        elif e['kind'] == 'set_transform':
            for key in ['zoom', 'pan_x', 'pan_y']:
                frame[key] = e[key]
            frame['zoom'] = min(max_zoom, max(min_zoom, frame['zoom']))
        elif e['kind'] == 'new_data':
            frame['series'] = e['series']
        elif e['kind'] == 'clear_selection':
            brushes = []
        elif e['kind'] == 'brush':
            brushes.append({'frame': frame.copy(), 'box': [min(e['x0'], e['x1']), min(e['y0'], e['y1']),
                                                        max(e['x0'], e['x1']), max(e['y0'], e['y1'])], 'mode': e['mode']})
    selected = []
    for brush in brushes:
        points = _projection(brush['frame'] if bits[5] else frame, bits, pan_scales)
        box = brush['box']
        hits = []
        for key, line in points.items():
            hit = any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in line)
            if bits[4]:
                hit = hit or any(_segment_hit(a, b, box) for a, b in zip(line, line[1:]))
            if hit:
                hits.append(key)
        mode = brush['mode']
        if mode == 'replace':
            selected = hits[:]
        elif mode == 'add':
            selected = sorted(selected + [k for k in hits if k not in selected])
        elif mode == 'subtract':
            selected = [k for k in selected if k not in hits]
        else:
            selected = sorted([k for k in selected if k not in hits] + [k for k in hits if k not in selected])
    return {'projected': _projection(frame, bits, pan_scales), 'selected_ids': sorted(selected),
            'linked_counts': [int(key in selected) for key in sorted(s['series'])]}


def _physics(s, bits, negative_cap, positive_cap, cap_before_gravity):
    b = s['initial_body'].copy()
    gravity, dt, contacts, grounded, contact = 0.0, 1/60, 0, False, None
    floor, wall = s['floor'], s['wall']
    for e in s['events']:
        kind = e['kind']
        if kind == 'reset_scene':
            b = s['initial_body'].copy()
            gravity, dt, contacts, grounded, contact = 0.0, 1/60, 0, False, None
        elif kind == 'set_body':
            for key in ['x', 'y', 'vx', 'vy']:
                b[key] = e[key]
            grounded, contact = False, None
        elif kind == 'set_physics':
            gravity, dt = e['gravity'], e['dt']
        elif kind == 'advance':
            for _ in range(e['ticks']):
                grounded, contact = False, None
                if cap_before_gravity:
                    b['vy'] = min(positive_cap, max(-negative_cap, b['vy']))
                b['vy'] += gravity * [1, 1.5][bits[4]] * dt
                if not cap_before_gravity:
                    b['vy'] = min(positive_cap, max(-negative_cap, b['vy']))
                b['vx'] *= max(0, 1 - [0, .04][bits[3]] * dt * 60)
                nx, ny = b['x'] + b['vx'] * dt, b['y'] + b['vy'] * dt
                if (b['vx'] > 0 and b['x'] + b['width'] <= wall['x'] <= nx + b['width']
                        and b['y'] + b['height'] > wall['y'] and b['y'] < wall['y'] + wall['height']):
                    nx = wall['x'] - b['width']
                    b['vx'] *= -[0, 1][bits[5]]
                    contacts += 1
                    contact = 'wall'
                overlap = nx + b['width'] > floor['x'] and nx < floor['x'] + floor['width']
                if overlap and b['vy'] > 0 and b['y'] + b['height'] <= floor['y'] <= ny + b['height']:
                    ny = floor['y'] - b['height']
                    b['vy'] *= -[0, .6][bits[1]]
                    b['vx'] *= 1 - [0, .3][bits[2]]
                    contacts += 1
                    contact, grounded = 'floor', b['vy'] == 0
                elif (overlap and not bits[0] and b['vy'] < 0
                      and b['y'] >= floor['y'] + floor['height'] >= ny):
                    ny = floor['y'] + floor['height']
                    b['vy'] = abs(b['vy']) * [0, .6][bits[1]]
                    contacts += 1
                    contact = 'floor'
                if overlap and b['vy'] == 0 and abs(ny + b['height'] - floor['y']) < 1e-7:
                    grounded, contact = True, 'floor'
                b['x'] = nx
                b['y'] = ny
    return {'x': b['x'], 'y': b['y'], 'vx': b['vx'], 'vy': b['vy'],
            'grounded': grounded, 'contact': contact, 'contacts': contacts}


FUNCTIONS = {
    'drag_hysteresis': (_drag,),
    'linked_brush': (_projection, _segment_hit, _chart),
    'contact_dynamics': (_physics,),
}


def expected(family, state, mask, split='train'):
    args = parameters(family, mask, split)
    bits = [(mask >> i) & 1 for i in range(6)]
    return FUNCTIONS[family][-1](state, bits, *args)


def oracle_source(family, mask, split='train'):
    args = parameters(family, mask, split)
    bits = [(mask >> i) & 1 for i in range(6)]
    body = '\n\n'.join(inspect.getsource(f) for f in FUNCTIONS[family])
    values = ', '.join(repr(value) for value in args)
    return (body + f'\n\ndef solve(state):\n    return '
            f'{FUNCTIONS[family][-1].__name__}(state, {bits!r}, {values})\n')


def _state(family, seed, events):
    state = initial_state(family, seed)
    for event in events:
        state = transition(family, state, event)
    return state


def protected_states(family, seed, split='train', pool='final'):
    """Target-independent histories covering boundaries and interacting behavior.

    Progress and final use distinct RNG streams, scene geometries and numeric
    probes. Drag's 18 exact threshold cases and reset invariant intentionally
    test the same behavior; other drag stimuli differ. Disjoint histories do
    not mean fully disjoint behavioral coverage or boundary values. Pools
    contain no actor instructions.
    """
    if pool not in ('progress', 'final'):
        raise ValueError('Unknown grading pool')
    parameters(family, 0, split)
    salt = 104729 if pool == 'progress' else 130363
    rng = random.Random(seed * 1000003 + salt)
    scene_seed = rng.randrange(1_000_000_000)
    # Public transform/body prefixes vary across pools without exposing a pool
    # label to candidate code. The tests themselves remain fully replayable.
    cases = []

    def add(events):
        cases.append(_state(family, scene_seed, events))

    if family == 'drag_hysteresis':
        z = rng.uniform(.82, 1.18)
        prefix = [dict(kind='set_zoom', zoom=z)]
        for width in (439, 440, 441, 479, 480, 481, 519, 520, 521):
            add(prefix + [dict(kind='resize', width=width)])
            add(prefix + [dict(kind='resize', width=700),
                          dict(kind='resize', width=width)])
        # Distinct non-boundary progress stimuli exercise other cards and
        # scales. A prefix later overwritten by set_zoom would not provide
        # meaningful separation from the final pool.
        zooms = ((.6, .9, 1.1, 1.4, 1.6, 1.9) if pool == 'progress'
                 else (.5, .7, 1, 1.3, 1.8, 2))
        for zoom in zooms:
            for wide in (False, True):
                if pool == 'progress':
                    card = rng.randrange(1, 4)
                    x = 85 + card * 130 if wide else 105 + (card % 2) * 140
                    y = 305 if wide else 255 + (card // 2) * 110
                    dx, dy = rng.uniform(22, 28), rng.uniform(14, 19)
                else:
                    x, y = ((85, 305) if wide else (105, 255))
                    dx, dy = 17, 23
                pre = prefix + ([dict(kind='resize', width=640)] if wide else [])
                pre += [dict(kind='set_zoom', zoom=zoom), dict(kind='down', x=x, y=y)]
                add(pre)
                moved = pre + [dict(kind='move', x=x+dx, y=y+dy)]
                add(moved)
                add(moved + [dict(kind='up')])
        for outside in (False, True):
            for dx in (7, 19, 34):
                ox, oy = rng.uniform(85, 115), rng.uniform(240, 260)
                if pool == 'progress':
                    move_dx = rng.uniform(230, 240) if outside else dx + rng.uniform(2, 4)
                    move_dy = rng.uniform(17, 21)
                else:
                    move_dx, move_dy = (220 if outside else dx), 13
                move = dict(kind='move', x=ox+move_dx, y=oy+move_dy)
                seq = prefix + [dict(kind='down', x=ox, y=oy), move]
                add(seq)
                add(seq + [dict(kind='up'), dict(kind='set_zoom', zoom=1.7),
                           dict(kind='resize', width=700), dict(kind='resize', width=500)])
        # Held reflow, zoom change, release, then a fresh drag: lift must remain
        # output-only and neither offsets nor hit testing may accumulate it.
        if pool == 'progress':
            add(prefix + [dict(kind='down', x=245, y=255),
                          dict(kind='set_zoom', zoom=1.6), dict(kind='resize', width=640),
                          dict(kind='move', x=252, y=295)])
            add(prefix + [dict(kind='down', x=245, y=255), dict(kind='move', x=268, y=272),
                          dict(kind='up'), dict(kind='down', x=250, y=260)])
        else:
            add(prefix + [dict(kind='down', x=105, y=255),
                          dict(kind='set_zoom', zoom=1.8), dict(kind='resize', width=640),
                          dict(kind='move', x=112, y=263)])
            add(prefix + [dict(kind='down', x=105, y=255), dict(kind='move', x=119, y=269),
                          dict(kind='up'), dict(kind='down', x=110, y=260)])
        add(prefix + [dict(kind='down', x=105, y=255), dict(kind='reset_scene')])
    elif family == 'linked_brush':
        bounds = PARAMETERS[split][family]
        zooms = (.5, 2, 1) + bounds[0] + bounds[1]
        pan_x, pan_y = rng.uniform(12, 32), rng.uniform(-28, -8)
        for context in ('overview', 'A', 'B'):
            for zoom in zooms:
                pre = [dict(kind='set_view', context=context),
                       dict(kind='set_transform', zoom=zoom, pan_x=pan_x, pan_y=pan_y)]
                add(pre)
                brush = dict(kind='brush', x0=115, y0=215, x1=435, y1=525, mode='replace')
                add(pre + [brush, dict(kind='set_transform', zoom=2.5-zoom,
                                      pan_x=-pan_x, pan_y=-pan_y)])
        for mode in ('replace', 'add', 'subtract', 'toggle'):
            for context in ('overview', 'A', 'B'):
                seq = [dict(kind='set_transform', zoom=.5, pan_x=pan_x, pan_y=pan_y),
                       dict(kind='brush', x0=100, y0=220, x1=350, y1=460, mode='replace'),
                       dict(kind='set_view', context=context),
                       dict(kind='set_transform', zoom=2, pan_x=-pan_x, pan_y=-pan_y),
                       dict(kind='brush', x0=435, y0=525, x1=115, y1=215, mode=mode)]
                add(seq)
                add(seq + [dict(kind='new_data', seed=rng.randrange(9999999))])
        for box in ((280, 330, 320, 370), (310, 350, 310, 350), (105, 420, 128, 470)):
            seq = [dict(kind='set_transform', zoom=1, pan_x=pan_x, pan_y=pan_y),
                   dict(kind='brush', x0=box[0], y0=box[1], x1=box[2], y1=box[3], mode='replace')]
            add(seq)
            add(seq + [dict(kind='clear_selection')])
        add([dict(kind='set_transform', zoom=.5, pan_x=pan_x, pan_y=pan_y),
             dict(kind='reset_scene')])
    else:
        neg, pos, _ = PARAMETERS[split][family]
        y = rng.uniform(170, 230)
        # Open space avoids collision masking speed/order differences.
        for vy in (-1000, -neg[0], -neg[1], neg[0], pos[0], pos[1], 1000):
            for gravity in (-900, 0, 900):
                for ticks in (1, 3):
                    add([dict(kind='set_body', x=0, y=y, vx=0, vy=vy),
                         dict(kind='set_physics', gravity=gravity, dt=.025),
                         dict(kind='advance', ticks=ticks)])
        for dt in (.01, .025, .05):
            for vx, vy, x, start_y in ((120, 1000, 200, 300), (0, -1000, 200, 430),
                                       (900, 0, 420, 250), (200, 300, 200, 180)):
                add([dict(kind='set_physics', gravity=rng.choice((-100, 150, 300)), dt=dt),
                     dict(kind='set_body', x=x+rng.uniform(-.5, .5), y=start_y, vx=vx, vy=vy),
                     dict(kind='advance', ticks=12)])
        for vy in (-1000, 1000):
            for gravity in (-900, 900):
                seq = [dict(kind='set_body', x=0, y=y, vx=0, vy=vy),
                       dict(kind='set_physics', gravity=gravity, dt=.05),
                       dict(kind='advance', ticks=2),
                       dict(kind='set_physics', gravity=-gravity, dt=.01)]
                add(seq + [dict(kind='advance', ticks=3)])
                add(seq + [dict(kind='reset_scene')])
        add([dict(kind='set_body', x=rng.uniform(190, 210), y=340, vx=0, vy=0),
             dict(kind='set_physics', gravity=0, dt=rng.uniform(.01, .04)),
             dict(kind='advance', ticks=3)])
    # Independent history composition probes. Never supplied as an actor plan.
    if family == 'drag_hysteresis':
        for i in (0, 1, 2, 3):
            x, y = 105 + (i % 2)*140, 255 + (i//2)*110
            seq = [dict(kind='down', x=x, y=y), dict(kind='move', x=x+8, y=y+29)]
            add(seq)
            add(seq + [dict(kind='up'), dict(kind='reset_scene'), dict(kind='down', x=x,y=y)])
    elif family == 'linked_brush':
        for zoom in (.51, 1.99):
            seq = [dict(kind='set_transform', zoom=zoom, pan_x=pan_x, pan_y=pan_y),
                   dict(kind='brush', x0=0,y0=100,x1=590,y1=550,mode='replace')]
            add(seq + [dict(kind='clear_selection')])
            add(seq + [dict(kind='reset_scene'),dict(kind='set_view',context='B')])
    else:
        for vy in (-500,-200,200,500):
            seq = [dict(kind='set_body',x=200,y=430 if vy<0 else 320,vx=0,vy=vy),
                   dict(kind='set_physics',gravity=0,dt=.05),dict(kind='advance',ticks=8)]
            add(seq)
            add(seq + [dict(kind='reset_scene')])
    if pool == 'final':
        # This composition grammar is absent from progress reward cases.
        for _ in range(12):
            if family == 'drag_hysteresis':
                card = rng.randrange(4)
                x,y=105+(card%2)*140,255+(card//2)*110
                seq=[dict(kind='down',x=x,y=y),dict(kind='set_zoom',zoom=rng.uniform(.5,2)),
                     dict(kind='resize',width=rng.choice((440,500,540,700))),
                     dict(kind='move',x=x+rng.uniform(-25,25),y=y+rng.uniform(-35,35))]
                add(seq);add(seq+[dict(kind='up'),dict(kind='reset_scene'),dict(kind='resize',width=500)])
            elif family == 'linked_brush':
                seq=[dict(kind='new_data',seed=rng.randrange(9999999)),
                     dict(kind='set_view',context=rng.choice(('A','B'))),
                     dict(kind='set_transform',zoom=rng.uniform(.5,2),pan_x=pan_x,pan_y=pan_y),
                     dict(kind='brush',x0=130,y0=230,x1=450,y1=510,mode='replace'),
                     dict(kind='new_data',seed=rng.randrange(9999999))]
                add(seq);add(seq+[dict(kind='brush',x0=160,y0=260,x1=410,y1=480,mode='toggle')])
            else:
                seq=[dict(kind='set_body',x=0,y=rng.uniform(100,200),vx=0,vy=rng.choice((-1100,1100))),
                     dict(kind='set_physics',gravity=rng.uniform(200,900),dt=.025),
                     dict(kind='advance',ticks=rng.randrange(1,5)),
                     dict(kind='set_physics',gravity=rng.uniform(-900,-200),dt=.05)]
                add(seq+[dict(kind='advance',ticks=2)])
                add(seq+[dict(kind='advance',ticks=3),dict(kind='reset_scene')])
    # Runtime cost is fixed per family/pool, independent of private world mask.
    return copy.deepcopy(cases)


# V4 adds safe shape constraints before rendering candidate outputs.
_base_valid = valid

def valid(family,state,value):
    if not _base_valid(family,state,value):
        return False
    if family == 'drag_hysteresis':
        return (all(p[2]>0 and p[3]>0 for p in value['positions'].values())
                and value['active_id'] in [None,*state['ids']]
                and value['layout'] in ('compact','wide'))
    if family == 'contact_dynamics':
        return value['contact'] in (None,'wall','floor') and type(value['contacts']) is int and value['contacts']>=0
    return all(type(x) is int and x in (0,1) for x in value['linked_counts'])


def panel(family,state,value,heading):
    from .rendering import panel as render
    return render(family,state,value,heading)
