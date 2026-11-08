"""Private Batch 4 references; never included in actor context or source.

The six original controlled factors are unchanged. Each family additionally has
one deterministic behavior shared by all 64 controlled worlds. These functions
are statically defined: no startup patching or mutation of module globals.
"""
import inspect

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
    # Output-only lift: replay/hit-testing and persistent offsets stay unchanged.
    if active is not None:
        positions[active][1] -= 8
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
            # Clamp when the transform is set, before any historical snapshot.
            frame['zoom'] = min(1.5, max(0.75, frame['zoom']))
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
                # Clamp each tick after gravity and before position integration.
                b['vy'] = max(-600, min(600, b['vy']))
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


FUNCTIONS = {'drag_hysteresis': (_drag,),
             'linked_brush': (_projection, _segment_hit, _chart),
             'contact_dynamics': (_physics,)}


def expected(family, state, mask):
    return FUNCTIONS[family][-1](state, [(mask >> i) & 1 for i in range(6)])


def oracle_source(family, mask):
    """Evaluator-only complete repair, compatible with restricted execution."""
    return '\n\n'.join(inspect.getsource(f) for f in FUNCTIONS[family]) + (
        f'\n\ndef solve(state):\n    return {FUNCTIONS[family][-1].__name__}'
        f'(state, {[(mask >> i) & 1 for i in range(6)]!r})\n')
