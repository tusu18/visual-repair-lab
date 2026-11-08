"""Pinned Pillow built-in scalable font; no host font fallback in v4."""
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .tasks import canonical,valid
@lru_cache(maxsize=32)
def font(size):
    return ImageFont.load_default(size=size)

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
        lines = [f"layout={v['layout']}  active={v['active_id']}"] + [f"{k}: x={p[0]:.1f}, y={p[1]:.1f}, w={p[2]:.1f}, h={p[3]:.1f}" for k, p in v['positions'].items()]
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
        size=14
        while size>6 and d.textlength(line,font=font(size))>568:
            size-=1
        d.text((16,578+i*22),line,font=font(size),fill='#142b48')
    return np.asarray(im)
