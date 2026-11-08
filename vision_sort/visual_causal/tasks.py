"""Public controls and private reference for three sequence-dependent repairs.

The functions in oracle_source are evaluator-only, never model input. Every
graded output field is drawn at the same precision used by the grader.
"""
import ast
import copy
import inspect
import json
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FAMILIES = ('drag_hysteresis', 'linked_brush', 'contact_dynamics')
FACTORS = {
    'drag_hysteresis': ('wide_threshold', 'hysteresis', 'pointer_capture', 'release_snap', 'x_axis_lock', 'zoom_conversion'),
    'linked_brush': ('A_x_power', 'A_y_reverse', 'B_x_power', 'B_y_reverse', 'segment_intersection', 'historical_frame'),
    'contact_dynamics': ('one_way', 'floor_bounce', 'contact_friction', 'air_drag', 'gravity_scale', 'wall_bounce'),
}


def _drag(s, bits):
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
                offsets[active] = [round(v / 20) * 20 for v in offsets[active]]
            active = None
        # Return the changed positions in this frame, not one event later.
        for i, key in enumerate(s['ids']):
            bx = 60 + (i % 2) * 140 if mode == 'compact' else 40 + i * 130
            by = 220 + (i // 2) * 110 if mode == 'compact' else 270
            positions[key] = [bx + offsets[key][0], by + offsets[key][1], 90, 70]
    return {'positions': positions, 'active_id': active, 'layout': mode}


def _projection(frame, bits):
    context, zoom = frame['context'], frame['zoom']
    power, reverse = 1, False
    if context != 'overview':
        off = 0 if context == 'A' else 2
        power, reverse = [1, 2][bits[off]], bool(bits[off + 1])
    out = {}
    for key, points in frame['series'].items():
        out[key] = [[70 + 460 * zoom * x ** power + frame['pan_x'],
                     200 + 300 * zoom * (y if reverse else 1 - y) + frame['pan_y']]
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


def _chart(s, bits):
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
        elif e['kind'] == 'new_data':
            frame['series'] = e['series']
        elif e['kind'] == 'clear_selection':
            brushes = []
        elif e['kind'] == 'brush':
            brushes.append({'frame': frame.copy(), 'box': [min(e['x0'], e['x1']), min(e['y0'], e['y1']),
                                                        max(e['x0'], e['x1']), max(e['y0'], e['y1'])], 'mode': e['mode']})
    selected = []
    for brush in brushes:
        points = _projection(brush['frame'] if bits[5] else frame, bits)
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
    return {'projected': _projection(frame, bits), 'selected_ids': sorted(selected),
            'linked_counts': [int(key in selected) for key in sorted(s['series'])]}


def _physics(s, bits):
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
                b['vy'] += gravity * [1, 1.5][bits[4]] * dt
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


FUNCTIONS = {'drag_hysteresis': (_drag,), 'linked_brush': (_projection, _segment_hit, _chart), 'contact_dynamics': (_physics,)}


def legacy_expected(family, state, mask):
    return FUNCTIONS[family][-1](state, [(mask >> i) & 1 for i in range(6)])


def legacy_oracle_source(family, mask):
    return '\n\n'.join(inspect.getsource(f) for f in FUNCTIONS[family]) + f'\n\ndef solve(state):\n    return {FUNCTIONS[family][-1].__name__}(state, {[(mask >> i) & 1 for i in range(6)]!r})\n'


def expected(family, state, mask):
    """The Batch 4 truth used consistently by grading, rendering and rewards."""
    from ..visual_batch4.reference import expected as batch4_expected
    return batch4_expected(family, state, mask)


def oracle_source(family, mask):
    from ..visual_batch4.reference import oracle_source as batch4_oracle_source
    return batch4_oracle_source(family, mask)


def starter_source(family):
    """Specialize the faulty baseline; do not hand the actor the oracle switches.

    This is semantically mask0, but dead branches, parameter arrays and reference
    variant selectors are removed. Repairs require actual code changes.
    """
    names = {f.__name__ for f in FUNCTIONS[family] if 'bits' in inspect.signature(f).parameters}
    class Specialize(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'bits':
                return ast.copy_location(ast.Constant(0), node)
            node = self.generic_visit(node)
            if isinstance(node.value, (ast.List, ast.Tuple)) and isinstance(node.slice, ast.Constant):
                return node.value.elts[node.slice.value]
            return node

        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
                return ast.copy_location(ast.Constant(not node.operand.value), node)
            return node

        def visit_If(self, node):
            node = self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test, ast.Constant) else node

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test, ast.Constant) else node

        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            node.args.args = [a for a in node.args.args if a.arg != 'bits']
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in names:
                node.args = node.args[:1]
            return node
    tree = Specialize().visit(ast.parse(legacy_oracle_source(family, 0)))
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def initial_state(family, seed):
    rng = random.Random(seed)
    s = {'family': family, 'events': []}
    if family == 'drag_hysteresis':
        s['ids'] = ['A', 'B', 'C', 'D']
    elif family == 'linked_brush':
        # Public geometry varies without disclosing private behavior settings.
        delta = rng.uniform(-.025, .025)
        s['series'] = {'S1': [[.1, .2+delta], [.9, .8+delta]], 'S2': [[.1, .8], [.9, .2]], 'S3': [[.2, .15], [.8, .15]]}
    else:
        s.update(initial_body={'x': 200.0, 'y': 180.0, 'vx': 0.0, 'vy': 0.0, 'width': 20, 'height': 20},
                 floor={'x': 120, 'y': 360, 'width': 360, 'height': 12},
                 wall={'x': 480, 'y': 120, 'width': 12, 'height': 380})
    return s


CONTROLS = {
    'drag_hysteresis': {'resize': {'width': [280, 800]}, 'set_zoom': {'zoom': [.5, 2]},
                        'down': {'x': [-500, 1500], 'y': [-500, 1500]},
                        'move': {'x': [-500, 1500], 'y': [-500, 1500]}, 'up': {}, 'reset_scene': {}},
    'linked_brush': {'set_view': {'context': ['overview', 'A', 'B']},
                     'set_transform': {'zoom': [.5, 2], 'pan_x': [-200, 200], 'pan_y': [-200, 200]},
                     'brush': {'x0': [0, 600], 'y0': [100, 600], 'x1': [0, 600], 'y1': [100, 600], 'mode': ['replace', 'add', 'subtract', 'toggle']},
                     'new_data': {'seed': [0, 9999999]}, 'clear_selection': {}, 'reset_scene': {}},
    'contact_dynamics': {'set_body': {'x': [-100, 550], 'y': [0, 650], 'vx': [-2000, 2000], 'vy': [-2000, 2000]},
                          'set_physics': {'gravity': [-1000, 1000], 'dt': [.005, .05]}, 'advance': {'ticks': [1, 60]}, 'reset_scene': {}},
}

# This distribution exposes the redacted Batch 4 contract. Legacy functions
# above remain solely to preserve the exact original public starter programs.
from ..visual_batch4.contracts import CONTRACTS


def transition(family, state, event):
    if type(event) is not dict or event.get('kind') not in CONTROLS[family]:
        raise ValueError('Unknown control')
    kind = event['kind']
    bounds = CONTROLS[family][kind]
    if set(event) != {'kind', *bounds}:
        raise ValueError('Control fields differ from contract')
    for key, choices in bounds.items():
        v = event[key]
        if isinstance(choices[0], str):
            if v not in choices:
                raise ValueError('Unknown option')
        elif type(v) not in (int, float) or not np.isfinite(v) or not choices[0] <= v <= choices[1]:
            raise ValueError('Control out of bounds')
    if kind in ('advance', 'new_data') and type(event['ticks' if kind == 'advance' else 'seed']) is not int:
        raise ValueError('Integer required')
    out = copy.deepcopy(state)
    e = copy.deepcopy(event)
    if kind == 'new_data':
        rng = random.Random(e.pop('seed'))
        e['series'] = {key: [[rng.uniform(.08, .92), rng.uniform(.08, .92)] for _ in range(3)] for key in state['series']}
    out['events'].append(e)
    return out


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if type(value) in (float, int):
        return round(float(value), 1)
    return value


def valid(family, state, value):
    # Only schema/finite-number validity, independent of the private target.
    if type(value) is not dict:
        return False
    try:
        json.dumps(value, allow_nan=False)
        shape = expected(family, state, 0)
        def check(a, b):
            if type(b) is dict:
                return type(a) is dict and set(a) == set(b) and all(check(a[k], b[k]) for k in b)
            if type(b) is list:
                if type(a) is not list:
                    return False
                if family == 'linked_brush' and b is shape['selected_ids']:
                    return len(a) == len(set(a)) and all(k in state['series'] for k in a)
                return len(a) == len(b) and all(check(x, y) for x, y in zip(a, b))
            if type(b) is bool:
                return type(a) is bool
            if type(b) in (int, float):
                return type(a) in (int, float) and abs(a) < 1e7
            return a is None or isinstance(a, str)
        return check(value, shape)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


@lru_cache(maxsize=8)
def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(family, state, value, heading):
    im = Image.new('RGB', (600, 700), '#f3f6fc')
    d = ImageDraw.Draw(im)
    d.text((20, 18), heading, font=font(22), fill='#142b48')
    d.text((20, 56), family.replace('_', ' '), font=font(18), fill='#526783')
    d.text((20, 90), f"{len(state['events'])} public events · persistent state", font=font(15), fill='#526783')
    d.rectangle((20, 130, 580, 560), outline='#a1b4ce', width=2)
    if value is None or not valid(family, state, value):
        d.text((40, 260), 'No schema-valid candidate output', font=font(20), fill='#b33737')
        return np.asarray(im)
    v = canonical(value)
    if family == 'drag_hysteresis':
        for key, (x, y, w, h) in v['positions'].items():
            d.rectangle((x, y, x+w, y+h), fill='#c5dcff', outline='#2161bb', width=3)
            d.text((x+35, y+20), key, font=font(24), fill='#142b48')
        lines = [f"layout={v['layout']}  active={v['active_id']}"] + [f"{k}: x={p[0]:.1f}, y={p[1]:.1f}" for k, p in v['positions'].items()]
    elif family == 'linked_brush':
        colors = ['#2366c5', '#d56b21', '#258c5a']
        for i, (key, line) in enumerate(v['projected'].items()):
            if len(line) > 1:
                d.line([tuple(p) for p in line], fill=colors[i], width=4 if key in v['selected_ids'] else 2)
            for x, y in line:
                d.ellipse((x-4, y-4, x+4, y+4), fill=colors[i])
            d.text((70+i*160, 160), key, fill=colors[i], font=font(18))
        # All graded projected coordinates are printed; evidence is never hidden
        # in a field that cannot affect rendered output.
        lines = ['selected=' + ','.join(v['selected_ids']) + ' counts=' + str(v['linked_counts'])]
        for key, pts in v['projected'].items():
            lines.append(key + ': ' + ' '.join(f'({x:.1f},{y:.1f})' for x, y in pts))
    else:
        f, w = state['floor'], state['wall']
        for p in (f, w):
            d.rectangle((p['x'], p['y'], p['x']+p['width'], p['y']+p['height']), fill='#9baec3')
        x, y = v['x'], v['y']
        d.rectangle((x, y, x+20, y+20), fill='#2a6ed0')
        lines = [f"x={x:.1f} y={y:.1f}", f"vx={v['vx']:.1f} vy={v['vy']:.1f}",
                 f"grounded={v['grounded']} contact={v['contact']}", f"contacts={v['contacts']:.0f}"]
    # Repaint footer so a plot outside normal bounds cannot cover state text.
    d.rectangle((0, 570, 600, 700), fill='#f3f6fc')
    for i, line in enumerate(lines):
        d.text((16, 578+i*22), line, font=font(14), fill='#142b48')
    return np.asarray(im)


def baseline_events(family):
    if family == 'drag_hysteresis':
        return [dict(kind='resize', width=w) for w in (500, 570, 500, 460, 500)] + [
            dict(kind='reset_scene'), dict(kind='set_zoom', zoom=1.5), dict(kind='down', x=105, y=255),
            dict(kind='move', x=122, y=280), dict(kind='up'), dict(kind='reset_scene'),
            dict(kind='down', x=105, y=255), dict(kind='move', x=300, y=300), dict(kind='up')]
    if family == 'linked_brush':
        return [dict(kind='set_view', context=c) for c in ('A', 'B')] + [dict(kind='reset_scene'),
            dict(kind='brush', x0=280, y0=330, x1=320, y1=370, mode='replace'), dict(kind='clear_selection'),
            dict(kind='brush', x0=105, y0=420, x1=128, y1=470, mode='replace'),
            dict(kind='set_transform', zoom=2, pan_x=0, pan_y=0)]
    return [dict(kind='set_body', x=200, y=200, vx=80, vy=500), dict(kind='advance', ticks=24),
            dict(kind='reset_scene'), dict(kind='set_body', x=200, y=430, vx=0, vy=-500), dict(kind='advance', ticks=16),
            dict(kind='reset_scene'), dict(kind='set_body', x=200, y=180, vx=200, vy=0),
            dict(kind='set_physics', gravity=200, dt=1/60), dict(kind='advance', ticks=12),
            dict(kind='reset_scene'), dict(kind='set_body', x=420, y=250, vx=500, vy=0), dict(kind='advance', ticks=10)]


def protected_states(family, seed):
    """Independent parameterized sequences, never supplied as agent test steps."""
    rng = random.Random(seed)
    states = [initial_state(family, seed)]
    s = states[0]
    for e in baseline_events(family):
        s = transition(family, s, e)
        states.append(s)
    for _ in range(12):
        s = initial_state(family, rng.randrange(1_000_000))
        if family == 'drag_hysteresis':
            events = [dict(kind='resize', width=rng.choice([479, 481, 499, 521, 559])), dict(kind='resize', width=640), dict(kind='resize', width=rng.choice([459, 481, 501])), dict(kind='reset_scene'), dict(kind='set_zoom', zoom=rng.choice([.7, 1.3, 1.8])), dict(kind='down', x=105, y=255), dict(kind='move', x=105+rng.randrange(-20, 150), y=255+rng.randrange(-20, 150)), dict(kind='up')]
        elif family == 'linked_brush':
            events = [dict(kind='set_view', context=rng.choice(['overview', 'A', 'B'])), dict(kind='brush', x0=rng.randrange(80, 300), y0=rng.randrange(210, 400), x1=rng.randrange(310, 550), y1=rng.randrange(410, 550), mode=rng.choice(['replace', 'add', 'subtract', 'toggle'])), dict(kind='set_transform', zoom=rng.choice([.8, 1.4, 2]), pan_x=20, pan_y=-10), dict(kind='brush', x0=110, y0=220, x1=430, y1=520, mode=rng.choice(['add', 'subtract', 'toggle'])), dict(kind='new_data', seed=rng.randrange(9999999))]
        else:
            events = [dict(kind='set_body', x=rng.randrange(140, 430), y=rng.choice([180, 330, 410]), vx=rng.choice([-100, 100, 800]), vy=rng.choice([-900, 0, 900])), dict(kind='set_physics', gravity=rng.choice([0, 100, 500]), dt=rng.choice([.01, .025, .05])), dict(kind='advance', ticks=rng.randrange(2, 35))]
        for e in events:
            s = transition(family, s, e)
        states.append(s)
    return states
