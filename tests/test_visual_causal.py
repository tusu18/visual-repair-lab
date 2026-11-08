"""Observable truth, sequence dependence, private inputs and repair semantics."""
import copy
import json

import numpy as np
import pytest

from vision_sort.visual_causal import VisualCausalEnv
from vision_sort.visual_causal.tasks import (FAMILIES, baseline_events, canonical, expected,
    initial_state, legacy_expected, oracle_source, panel, protected_states, starter_source, transition, valid)
from vision_sort.visual_discovery.runtime import run_program


def interact(env, event):
    return env.step({'type': 'interact', 'event': event})


@pytest.mark.parametrize('family', FAMILIES)
def test_all_worlds_have_identical_initial_visible_output(family):
    s = initial_state(family, 1810000)
    neutral = canonical(expected(family, s, 0))
    assert all(canonical(expected(family, s, mask)) == neutral for mask in range(64))
    a, b = VisualCausalEnv(family), VisualCausalEnv(family)
    oa, ia = a.reset(seed=1810000, options={'world_seed': 1810010})
    ob, ib = b.reset(seed=1810000, options={'world_seed': 1810011})
    assert oa['context'] == ob['context'] and ia == ib
    assert np.array_equal(oa['image'], ob['image'])
    assert a.observation_space.contains(oa)
    assert a.entropy == 6
    assert not {'world_seed', 'mask', 'posterior', 'claims', 'metrics'} & set(json.loads(oa['context']))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('world_seed', [1810020, 1810021, 1810022])
def test_handwritten_public_schedule_identifies_six_factors(family, world_seed):
    env = VisualCausalEnv(family)
    env.reset(seed=1810001, options={'world_seed': world_seed})
    for event in baseline_events(family):
        interact(env, event)
    report = env.research_report()
    assert env.entropy == 0 and report['information_bits'] == 6
    assert len(report['mutants_killed']) == 6
    assert report['rewarded_mutants_killed'] == 4 and report['reward_held_out_mutants_killed'] == 2


@pytest.mark.parametrize('family', FAMILIES)
def test_bounded_reference_and_specialized_starter_parity(family):
    assert 'bits' not in starter_source(family)
    for state in protected_states(family, 1810002):
        for mask in (0, 21, 42, 63):
            r = run_program(oracle_source(family, mask), state)
            assert r['ok'], r['error']
            assert valid(family, state, r['result'])
            assert canonical(r['result']) == canonical(expected(family, state, mask))
        starter = run_program(starter_source(family), state)
        # Batch 4 intentionally preserves the original starter while revising truth.
        assert starter['ok'] and canonical(starter['result']) == canonical(legacy_expected(family, state, 0))


def test_hysteresis_needs_history_at_identical_current_width():
    s = initial_state('drag_hysteresis', 1810003)
    a = transition('drag_hysteresis', s, {'kind': 'resize', 'width': 500})
    b = transition('drag_hysteresis', s, {'kind': 'resize', 'width': 570})
    b = transition('drag_hysteresis', b, {'kind': 'resize', 'width': 500})
    assert expected('drag_hysteresis', a, 3)['layout'] == 'compact'
    assert expected('drag_hysteresis', b, 3)['layout'] == 'wide'


def test_downward_motion_does_not_identify_one_way_floor():
    f = 'contact_dynamics'
    s = initial_state(f, 1810004)
    s = transition(f, s, dict(kind='set_body', x=200, y=200, vx=0, vy=500))
    s = transition(f, s, dict(kind='advance', ticks=24))
    assert expected(f, s, 0) == expected(f, s, 1)
    s = initial_state(f, 1810004)
    s = transition(f, s, dict(kind='set_body', x=200, y=430, vx=0, vy=-500))
    s = transition(f, s, dict(kind='advance', ticks=16))
    assert expected(f, s, 0) != expected(f, s, 1)


@pytest.mark.parametrize('family', FAMILIES)
def test_correct_repair_can_submit_without_verification_quota(family):
    env = VisualCausalEnv(family, max_steps=2)
    env.reset(seed=1810005, options={'world_seed': 1810030})
    env.step({'type': 'edit', 'source': oracle_source(family, env._mask)})
    _, _, term, trunc, _ = env.step({'type': 'submit'})
    assert term and not trunc and env.final_result['success']
    assert not env.research_report()['latest_revision_tests']


def test_latest_revision_evidence_and_crop_information():
    env = VisualCausalEnv('linked_brush')
    env.reset(seed=1810006, options={'world_seed': 1810031})
    for event in baseline_events(env.family):
        interact(env, event)
    before = env.entropy
    env.step({'type': 'crop', 'box': [0, 100, 600, 600]})
    assert env.entropy == before and env.research_report()['steps'][-1]['information_gain'] == 0
    env.step({'type': 'edit', 'source': env.source+'\n# revision1\n'})
    env.step({'type': 'observe'})
    assert not env.research_report()['latest_revision_tests']
    env.step({'type': 'run'})
    assert env.research_report()['latest_revision_tests']
    env.step({'type': 'edit', 'source': 'def solve(state):\n return {}\n'})
    assert not env.research_report()['latest_revision_tests']
    env.step({'type': 'run'})
    assert not env.research_report()['latest_revision_tests']


def test_source_only_zero_images_and_no_visual_credit_or_seeing_claim():
    env = VisualCausalEnv('linked_brush', image_access=False)
    obs, _ = env.reset(seed=1810007, options={'world_seed': 1810032})
    for event in baseline_events(env.family):
        obs, *_ = interact(env, event)
    env.record_claims([{'path': '/selected_ids', 'value_json': '[]'}])
    assert not obs['image'].any() and not env.render().any()
    r = env.research_report()
    assert r['information_bits'] == 0 and not r['mutants_killed']
    assert r['incorrect_visible_claims'] == r['correct_visible_claims'] == 0


def test_claims_score_only_explicit_visible_facts_without_feedback():
    env = VisualCausalEnv('drag_hysteresis')
    before, _ = env.reset(seed=1810008, options={'world_seed': 1810033})
    env.record_claims([{'path': '/layout', 'value_json': '"compact"'},
                       {'path': '/active_id', 'value_json': '"A"'},
                       {'path': '/private_factor', 'value_json': 'true'}])
    after = env._observation()
    assert before['context'] == after['context'] and np.array_equal(before['image'], after['image'])
    r = env.research_report()
    assert r['correct_visible_claims'] == r['incorrect_visible_claims'] == 1
    assert not r['claims'][2]['valid_path']


@pytest.mark.parametrize('truncate', [False, True])
def test_potential_shaping_terminal_zero_and_truncation_bootstrap(truncate):
    env = VisualCausalEnv('linked_brush', objective='investigate', reward_mode='train', gamma=.9, max_steps=1 if truncate else 4)
    env.reset(seed=1810009, options={'world_seed': 1810034})
    _, reward, term, trunc, _ = interact(env, dict(kind='set_view', context='A'))
    assert reward == pytest.approx(-.025+.9*(2/6)) and trunc == truncate and not term
    if not truncate:
        env.step({'type': 'submit'})
        rows = env.research_report()['steps']
        assert rows[0]['shaping_reward']+.9*rows[1]['shaping_reward'] == pytest.approx(0)


def test_reward_held_out_mechanisms_do_not_change_potential():
    env = VisualCausalEnv('linked_brush', objective='test_design', reward_mode='train')
    env.reset(seed=1810010, options={'world_seed': 1810035})
    before = env._potential()
    env._kills |= {'segment_intersection', 'historical_frame'}
    assert env._potential() == before


def test_bad_actions_consume_budget_and_cannot_leak_world_in_text():
    envs = [VisualCausalEnv('drag_hysteresis', max_steps=2) for _ in range(2)]
    for i, e in enumerate(envs):
        e.reset(seed=1810011, options={'world_seed': 1810040+i})
    for action in ['not json', {'type': 'interact', 'event': {'kind': 'resize', 'width': -1}}]:
        a = envs[0].step(copy.deepcopy(action))
        b = envs[1].step(copy.deepcopy(action))
        assert a[0]['context'] == b[0]['context'] and a[1:] == b[1:]
    assert a[3] and envs[0].steps == 2


@pytest.mark.parametrize('family', FAMILIES)
def test_every_final_output_change_is_visible_in_neutral_panel(family):
    state = initial_state(family, 1810012)
    value = expected(family, state, 0)
    if family == 'contact_dynamics':
        changes = dict(x=220, y=210, vx=10, vy=10, grounded=True, contact='floor', contacts=1)
        for key, replacement in changes.items():
            changed = copy.deepcopy(value)
            changed[key] = replacement
            assert not np.array_equal(panel(family, state, value, 'same'), panel(family, state, changed, 'same'))
    else:
        changed = copy.deepcopy(value)
        if family == 'drag_hysteresis':
            changed['active_id'] = 'A'
        else:
            changed['selected_ids'] = ['S1']
        assert not np.array_equal(panel(family, state, value, 'same'), panel(family, state, changed, 'same'))
