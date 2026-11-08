"""Reviewed hook compiler. Changing operation order changes the program itself."""
from vision_sort.visual_training.tasks import oracle_source
from .dsl import validate


def condition(family,name):
    scopes={
        'drag_hysteresis': {'always':'True','wide':"mode == 'wide'",'compact':"mode == 'compact'",'zoom_high':'zoom > 1.5','range':'1 <= zoom <= 1.5'},
        'linked_brush': {'always':'True','view_a':"frame['context'] == 'A'",'view_b':"frame['context'] == 'B'",'zoom_high':'zoom > 1','range':'150 <= x <= 450'},
        'contact_dynamics': {'always':'True','positive':"b['vy'] > 0",'negative':"b['vy'] < 0",'gravity_positive':'gravity > 0','range':"100 <= abs(b['vy']) <= 600"},
    }
    return scopes[family][name]


def replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise RuntimeError('Reviewed compiler anchor mismatch')
    return source.replace(old,new)


def compile_source(spec,disabled=(),mask=None):
    validate(spec)
    family=spec['family'];base=spec['world']['mask'] if mask is None else mask
    source=oracle_source(family,base,'train')
    rules=[r for i,r in enumerate(spec['rules']) if i not in disabled]
    if family=='linked_brush':
        # Only _projection's return is replaced. All brush/history logic uses it.
        marker='    return out\n'
        code=[]
        for rule in rules:
            op,value=rule['op'],repr(rule['value']);guard=condition(family,rule['condition'])
            expression={
                'limit':f'x = min({value}, max(70, x))',
                'gain':f'x = 70 + (x - 70) * {value}',
                'threshold':f'x = 70 if x < {value} else x',
                'snap':f'x = round(x / {value}) * {value}',
                'offset':f'y += {value}',
                'order':f'x = 70 + round((x - 70) / zoom / {value}) * {value} * zoom',
            }[op]
            code.append(f'                if {guard}:\n                    {expression}\n')
        if code:
            block="    for key in out:\n        line = []\n        for x, y in out[key]:\n"+''.join(code)+"                line.append([x, y])\n        out[key] = line\n"
            source=replace(source,marker,block+marker)
    elif family=='drag_hysteresis':
        for rule in rules:
            op,value=rule['op'],repr(rule['value']);guard=condition(family,rule['condition'])
            if op=='limit':
                anchor="            positions[key] = [bx + offsets[key][0], by + offsets[key][1], 90, 70]"
                extra=f"\n            if {guard}:\n                positions[key][0] = min(width - 90 - {value}, max({value}, positions[key][0]))"
                source=replace(source,anchor,anchor+extra,count=2)
            elif op=='gain':
                for axis in ('x','y'):
                    old=f"(e['{axis}'] - origin[{0 if axis=='x' else 1}]) / divisor"
                    source=replace(source,old,f'({old}) * ({value} if {guard} else 1)')
            elif op=='threshold':
                anchor='            high = [480, 520][bits[0]]'
                source=replace(source,anchor,anchor+f'\n            if {guard}:\n                high = {value}')
            elif op=='snap':
                anchor="        if kind == 'up' and active is not None:"
                source=replace(source,anchor,anchor+f'\n            if {guard}:\n                offsets[active] = [round(v / {value}) * {value} for v in offsets[active]]')
            elif op=='offset':
                anchor='        positions[active][1] -= lift * (zoom if lift_scales else 1)'
                source=replace(source,anchor,anchor+f'\n        if {guard}:\n            positions[active][1] -= {value}')
            else:
                anchor="        if kind == 'up' and active is not None:"
                source=replace(source,anchor,f"        if kind == 'move' and active is not None and ({guard}):\n            offsets[active][0] = round(offsets[active][0] * zoom / {value}) * {value} / zoom\n"+anchor)
    else:
        for rule in rules:
            op,value=rule['op'],repr(rule['value']);guard=condition(family,rule['condition'])
            if op=='gain':
                anchor="                b['vy'] += gravity * [1, 1.5][bits[4]] * dt"
                source=replace(source,anchor,anchor+f"\n                if {guard}:\n                    b['vy'] += gravity * [1, 1.5][bits[4]] * dt * ({value} - 1)")
            elif op=='order':
                anchor='                if cap_before_gravity:'
                source=replace(source,anchor,f'                step_cap_before_gravity = {value} if {guard} else cap_before_gravity\n                if step_cap_before_gravity:')
                source=replace(source,'                if not cap_before_gravity:','                if not step_cap_before_gravity:')
            elif op in ('limit','threshold'):
                anchor="                b['vx'] *= max(0, 1 - [0, .04][bits[3]] * dt * 60)"
                expression=(f"b['vx'] = min({value}, max(-{value}, b['vx']))" if op=='limit' else
                            f"b['vy'] = 0 if 0 < b['vy'] < {value} else b['vy']")
                source=replace(source,anchor,f'                if {guard}:\n                    {expression}\n'+anchor)
            else:
                anchor="                nx, ny = b['x'] + b['vx'] * dt, b['y'] + b['vy'] * dt"
                expression=f'nx = round(nx / {value}) * {value}' if op=='snap' else f'ny += {value}'
                source=replace(source,anchor,anchor+f'\n                if {guard}:\n                    {expression}')
    return source
