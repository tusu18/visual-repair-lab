"""Evaluator-only structural implementation mutations, with held-out operators.

Reward-held-out operators are absent from test-power and comparison rewards.
They can still be exercised by independent correctness/progress cases. This is
not a claim of mechanism holdout from all supervision.
"""
import copy
from .tasks import oracle_source, canonical


def _replace(source, old, new, count=None):
    actual = source.count(old)
    if not actual or (count is not None and actual != count):
        raise RuntimeError('Mutation anchor changed')
    return source.replace(old, new)


def mutant_sources(family, mask, split):
    source = oracle_source(family, mask, split)
    train, heldout = {}, {}
    if family == 'linked_brush':
        clamp = "frame['zoom'] = min(max_zoom, max(min_zoom, frame['zoom']))"
        train['missing_zoom_clamp'] = _replace(source, clamp, "frame['zoom'] = frame['zoom']", 1)
        train['missing_lower_clamp'] = _replace(source, clamp, "frame['zoom'] = min(max_zoom, frame['zoom'])", 1)
        train['missing_upper_clamp'] = _replace(source, clamp, "frame['zoom'] = max(min_zoom, frame['zoom'])", 1)
        train['unscaled_pan'] = _replace(source, '(zoom if pan_scales else 1)', '1', 2)
        train['stale_selection'] = _replace(source, "elif e['kind'] == 'clear_selection':\n            brushes = []", "elif e['kind'] == 'clear_selection':\n            pass", 1)
        heldout['wrong_view_projection'] = _replace(source, "off = 0 if context == 'A' else 2", 'off = 0', 1)
        heldout['missing_segment_intersection'] = _replace(source, 'if bits[4]:', 'if False:', 1)
        heldout['compound_projection_and_clamp'] = _replace(heldout['wrong_view_projection'], clamp, "frame['zoom'] = frame['zoom']", 1)
    elif family == 'contact_dynamics':
        cap = "b['vy'] = min(positive_cap, max(-negative_cap, b['vy']))"
        train['missing_speed_caps'] = _replace(source, cap, "b['vy'] = b['vy']", 2)
        train['wrong_update_order'] = _replace(source, "    b = s['initial_body'].copy()\n    gravity", "    cap_before_gravity = not cap_before_gravity\n    b = s['initial_body'].copy()\n    gravity", 1)
        train['symmetric_cap'] = _replace(source, "    b = s['initial_body'].copy()\n    gravity", "    negative_cap = positive_cap\n    b = s['initial_body'].copy()\n    gravity", 1)
        train['wrong_restitution'] = _replace(source, '[0, .6]', '[0, .3]', 2)
        train['stale_reset_contacts'] = _replace(source, "gravity, dt, contacts, grounded, contact = 0.0, 1/60, 0, False, None\n        elif", "gravity, dt, grounded, contact = 0.0, 1/60, False, None\n        elif", 1)
        heldout['missing_underside_contact'] = _replace(source, 'overlap and not bits[0]', 'overlap and False', 1)
        heldout['wrong_grounded_report'] = _replace(source, "'grounded': grounded", "'grounded': False", 1)
        heldout['compound_cap_and_restitution'] = _replace(train['missing_speed_caps'], '[0, .6]', '[0, .3]', 2)
    else:
        lift = "positions[active][1] -= lift * (zoom if lift_scales else 1)"
        train['missing_lift'] = _replace(source, lift, 'pass', 1)
        train['unscaled_lift'] = _replace(source, '(zoom if lift_scales else 1)', '1', 1)
        train['lifted_capture_bounds'] = _replace(source, 'x, y, w, h = positions[active]', 'x, y, w, h = positions[active]\n            y -= lift * (zoom if lift_scales else 1)', 1)
        train['missing_release'] = _replace(source, "if kind == 'up' and active is not None:", "if False:", 1)
        train['stale_reset_offsets'] = _replace(source, "            offsets = {key: [0.0, 0.0] for key in s['ids']}", '            pass', 1)
        train['wrong_object_lift'] = _replace(source, lift, "positions[s['ids'][-1]][1] -= lift * (zoom if lift_scales else 1)", 1)
        heldout['missing_zoom_divisor'] = _replace(source, 'divisor = zoom if bits[5] else 1', 'divisor = 1', 1)
        heldout['missing_snap'] = _replace(source, 'if bits[3]:', 'if False:', 1)
        heldout['compound_capture_and_release'] = _replace(train['lifted_capture_bounds'], "if kind == 'up' and active is not None:", 'if False:', 1)
    return train, heldout


def compile_trusted(source):
    """Only evaluator-authored reference/mutant fixtures may enter this function."""
    namespace = {}
    exec(compile(source, '<trusted-mutant>', 'exec'), namespace)
    return namespace['solve']


def exposure(functions, state, truth):
    return {name for name, solve in functions.items()
            if canonical(solve(copy.deepcopy(state))) != canonical(truth)}
