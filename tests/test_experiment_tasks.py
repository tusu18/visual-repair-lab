"""Variable mechanisms, grader coverage and executable oracle parity.

Only development scenes in 3010000 onward are exercised. These are evaluator
unit tests, not model attempts or final held-out evaluation results.
"""
import copy
import hashlib
import json

import pytest

from vision_sort.visual_discovery.runtime import run_program
from vision_sort.visual_experiments.tasks import (
    CONTRACTS,
    FACTORS,
    FAMILIES,
    PARAMETERS,
    canonical,
    expected,
    initial_state,
    oracle_source,
    parameters,
    protected_states,
    starter_source,
    transition,
)


def after(family, events):
    state = initial_state(family, 3010000)
    for event in events:
        state = transition(family, state, event)
    return state


@pytest.mark.parametrize('family', FAMILIES)
def test_initial_output_and_starter_do_not_identify_private_world(family):
    state = initial_state(family, 3010000)
    assert all(expected(family, state, mask) == expected(family, state, 0)
               for mask in range(512))
    original = {
        'drag_hysteresis': '0a40bcfb06b530ee4098fb361552e18004f2ea6f2d65c8c8e25880df37c2eb42',
        'linked_brush': 'e02e836a2a15cc9d576ae72d407f3226e49c1ef18156ee5d922db95f44915b3b',
        'contact_dynamics': '4da341158c7db4cbcd6be27b2121ff2503c4e8aef2ca30855a63b57c9d5cc1db',
    }
    assert hashlib.sha256(starter_source(family).encode()).hexdigest() == original[family]
    assert len(FACTORS[family]) == 9
    for phrase in ('.55', '.8', '360', '660', 'lift_scales', 'snap_grid', 'Unknown'):
        assert phrase not in CONTRACTS[family]


@pytest.mark.parametrize('family', FAMILIES)
def test_train_validation_test_parameters_have_disjoint_numeric_values(family):
    train, val, test = (PARAMETERS[s][family] for s in ('train', 'validation', 'test'))
    for i in range(3):
        if isinstance(train[i][0], bool):
            continue
        assert set(train[i]).isdisjoint(val[i])
        assert set(train[i]).isdisjoint(test[i])
        assert set(val[i]).isdisjoint(test[i])
    for mask in (-1, 512, True, 1.0):
        with pytest.raises(ValueError):
            parameters(family, mask)


@pytest.mark.parametrize('family', FAMILIES)
def test_grading_pools_repeat_exactly_and_have_disjoint_histories(family):
    final = protected_states(family, 3010001, pool='final')
    progress = protected_states(family, 3010001, pool='progress')
    def encode(state):
        return json.dumps(state, sort_keys=True)
    assert 50 <= len(final) <= 90
    assert final == protected_states(family, 3010001, pool='final')
    assert len(set(map(encode, final))) == len(final)
    assert not set(map(encode, final)) & set(map(encode, progress))
    assert final != protected_states(family, 3010002, pool='final')


@pytest.mark.parametrize('family', FAMILIES)
def test_executable_oracle_matches_truth_over_full_development_pool(family):
    states = protected_states(family, 3010003)
    for mask in (0, 8, 63, 64, 127, 256, 511):
        source = oracle_source(family, mask)
        namespace = {}
        exec(source, namespace)
        for state in states:
            assert canonical(namespace['solve'](copy.deepcopy(state))) == canonical(
                expected(family, state, mask))
        # Bounded compatibility is checked on stateful multi-event cases.
        for state in (states[0], states[len(states)//2], states[-1]):
            actual = run_program(source, state)
            assert actual['ok'], actual.get('error')
            assert canonical(actual['result']) == canonical(expected(family, state, mask))


@pytest.mark.parametrize('family', FAMILIES)
def test_grader_can_expose_all_nine_factor_mutants(family):
    # All original factors enabled makes drag snapping's grid observable.
    mask = 63
    states = protected_states(family, 3010004)
    gold = [canonical(expected(family, s, mask)) for s in states]
    for i, factor in enumerate(FACTORS[family]):
        assert any(canonical(expected(family, s, mask ^ (1 << i))) != value
                   for s, value in zip(states, gold)), factor


def test_drag_lift_zoom_interaction_is_output_only_and_grid_requires_snap():
    f = 'drag_hysteresis'
    held = after(f, [dict(kind='set_zoom', zoom=1.5), dict(kind='down', x=105, y=255)])
    assert expected(f, held, 0)['positions']['A'][1] == 216
    assert expected(f, held, 64)['positions']['A'][1] == 208
    assert expected(f, held, 128)['positions']['A'][1] == 214
    assert expected(f, held, 64+128)['positions']['A'][1] == 202
    moved = transition(f, held, dict(kind='move', x=122, y=278))
    repeated = transition(f, moved, dict(kind='move', x=122, y=278))
    assert expected(f, moved, 192) == expected(f, repeated, 192)
    released = transition(f, moved, dict(kind='up'))
    assert expected(f, released, 8)['positions']['A'][:2] == [80, 240]
    assert expected(f, released, 8+256)['positions']['A'][:2] == [90, 250]
    assert expected(f, released, 0) == expected(f, released, 256)
    reset = transition(f, held, dict(kind='reset_scene'))
    assert expected(f, reset, 511) == expected(f, initial_state(f, 3010000), 511)


def test_chart_both_boundaries_pan_interaction_and_history():
    f = 'linked_brush'
    for mask in (0, 64, 128, 256, 511):
        lo, hi, scaled = parameters(f, mask)
        for requested, effective in ((.5, lo), (2, hi), (lo, lo), (hi, hi)):
            state = after(f, [dict(kind='set_transform', zoom=requested, pan_x=20, pan_y=-10)])
            result = expected(f, state, mask)
            x, y = state['series']['S1'][0]
            assert result['projected']['S1'][0] == pytest.approx([
                70 + 460 * effective * x + 20 * (effective if scaled else 1),
                200 + 300 * effective * (1-y) - 10 * (effective if scaled else 1),
            ])
    # Historical frame retains its clamped transform and scaled pan after a
    # change; a latest-frame implementation selects a different line set.
    state = after(f, [dict(kind='set_transform', zoom=2, pan_x=20, pan_y=-10),
                      dict(kind='brush', x0=140, y0=480, x1=180, y1=520, mode='replace'),
                      dict(kind='set_transform', zoom=.5, pan_x=-20, pan_y=10)])
    assert expected(f, state, 32)['selected_ids'] != expected(f, state, 0)['selected_ids']


def test_physics_signed_cap_order_after_gravity_and_each_tick():
    f = 'contact_dynamics'
    for mask in (0, 64, 128, 256, 511):
        neg, pos, before = parameters(f, mask)
        gravity_scale = 1.5 if mask & 16 else 1
        for initial, gravity in ((1000, 1000), (-1000, -1000)):
            state = after(f, [dict(kind='set_body', x=0, y=200, vx=0, vy=initial),
                              dict(kind='set_physics', gravity=gravity, dt=.025),
                              dict(kind='advance', ticks=2)])
            capped = pos if initial > 0 else -neg
            speed = capped + gravity * gravity_scale * .025 if before else capped
            result = expected(f, state, mask)
            assert result['vy'] == speed
            assert result['y'] == pytest.approx(200 + speed * .05)


def test_public_contracts_allow_presentation_offsets_and_transform_rules():
    drag = CONTRACTS['drag_hysteresis']
    chart = CONTRACTS['linked_brush']
    assert 'Down attempts to hold a card, resolving overlapping hits in reversed ID order.' in drag
    assert 'Move events while a card is held may update its offset' in drag
    assert 'Down hit-tests current cards' not in drag
    assert 'Set_zoom never changes how cards are drawn' not in drag
    assert 'Set_zoom changes the public zoom control.' in drag
    assert 'x=70+460*zoom*xvalue+pan_x' not in chart
    assert 'y=200+300*zoom*(1-yvalue)+pan_y' not in chart
    assert 'the view transform affects the projected coordinates.' in chart


def test_progress_drag_uses_different_cards_zooms_and_move_deltas():
    family = 'drag_hysteresis'
    progress = protected_states(family, 3010030, pool='progress')
    final = protected_states(family, 3010030, pool='final')
    progress_zooms, final_zooms = set(), set()
    for i in range(18, 54, 3):
        progress_zooms.add(next(e['zoom'] for e in reversed(progress[i]['events'])
                                if e['kind'] == 'set_zoom'))
        final_zooms.add(next(e['zoom'] for e in reversed(final[i]['events'])
                             if e['kind'] == 'set_zoom'))
        assert expected(family, progress[i], 0)['active_id'] in ('B', 'C', 'D')
        assert expected(family, final[i], 0)['active_id'] == 'A'
        down, move = progress[i+1]['events'][-2:]
        assert 22 <= move['x'] - down['x'] <= 28
        assert 14 <= move['y'] - down['y'] <= 19
    assert progress_zooms.isdisjoint(final_zooms)
    identical = []
    for i, (p_state, f_state) in enumerate(zip(progress, final)):
        if all(canonical(expected(family, p_state, mask)) ==
               canonical(expected(family, f_state, mask)) for mask in range(512)):
            identical.append(i)
    # Exact threshold probes and reset invariance are shared deliberately.
    assert identical == list(range(18)) + [68]


def test_progress_drag_still_exposes_every_relevant_factor_mutant():
    family = 'drag_hysteresis'
    states = protected_states(family, 3010030, pool='progress')
    for mask in (0, 63, 511):
        gold = [canonical(expected(family, state, mask)) for state in states]
        for i, factor in enumerate(FACTORS[family]):
            if factor == 'snap_grid' and not mask & 8:
                continue
            assert any(canonical(expected(family, state, mask ^ (1 << i))) != value
                       for state, value in zip(states, gold)), (mask, factor)
        for state in states[18:21]:
            bounded = run_program(oracle_source(family, mask), state)
            assert bounded['ok'], bounded.get('error')
            assert canonical(bounded['result']) == canonical(expected(family, state, mask))


def test_progress_revision_preserves_final_pool_witness_hashes():
    frozen = {'drag_hysteresis': 'a04f8e363d6d73d2a0107ad890c6ceef7915f03324ee958ef54c488d30533f7c', 'linked_brush': '47210ffaa9072073eb8d34b8133a0e0a24644d12565d9241d94475d13092ecdd', 'contact_dynamics': '47026f38eacef6f7643ef7d8e8e2a89f98be8470f667db1d9ec36524186ddd95'}
    for family in FAMILIES:
        states = protected_states(family, 3010030, pool='final')
        encoded = json.dumps(states, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(encoded).hexdigest() == frozen[family]
