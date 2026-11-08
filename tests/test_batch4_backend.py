"""Batch 4 truth consistency, exact starter preservation and private metrics."""
import copy
import hashlib
import json

import numpy as np
import pytest

from vision_sort.visual_batch4.env import VisualBatch4Env
from vision_sort.visual_batch4.reference import expected, oracle_source
from vision_sort.visual_causal.env import digest
from vision_sort.visual_causal.tasks import (CONTRACTS, FAMILIES, canonical,
    initial_state, legacy_expected, legacy_oracle_source, protected_states,
    starter_source, transition)
from vision_sort.visual_discovery.runtime import run_program
from vision_sort.visual_reward.env import RewardWeights

STARTER_HASHES = {
    'drag_hysteresis': '0a40bcfb06b530ee4098fb361552e18004f2ea6f2d65c8c8e25880df37c2eb42',
    'linked_brush': 'e02e836a2a15cc9d576ae72d407f3226e49c1ef18156ee5d922db95f44915b3b',
    'contact_dynamics': '4da341158c7db4cbcd6be27b2121ff2503c4e8aef2ca30855a63b57c9d5cc1db',
}


def world(family, mask):
    return next(i for i in range(10000) if int(digest([i, family]), 16) % 64 == mask)


def state_after(family, events):
    state = initial_state(family, 2890001)
    for event in events:
        state = transition(family, state, event)
    return state


def witness(family):
    if family == 'drag_hysteresis':
        events = [dict(kind='down', x=105, y=255)]
    elif family == 'linked_brush':
        events = [dict(kind='set_transform', zoom=2, pan_x=0, pan_y=0)]
    else:
        events = [dict(kind='set_body', x=0, y=0, vx=0, vy=1000), dict(kind='advance', ticks=1)]
    return state_after(family, events)


@pytest.mark.parametrize('family', FAMILIES)
def test_exact_original_starter_source_and_no_initial_target_leak(family):
    source = starter_source(family)
    assert hashlib.sha256(source.encode()).hexdigest() == STARTER_HASHES[family]
    state = initial_state(family, 2890001)
    assert all(expected(family, state, i) == expected(family, state, 0) for i in range(64))
    env = VisualBatch4Env(family=family)
    observation, _ = env.reset(seed=2890001, options={'world_seed': 9})
    context = json.loads(observation['context'])
    assert context['environment'] == 'VisualRepairBatch4-v0'
    assert context['budget'] == 16 and context['source'] == source
    assert env.weights == RewardWeights()
    assert not {'world_seed', 'mask', 'novel_witnesses', 'information_scope'} & context.keys()
    assert 'Unknown' not in CONTRACTS[family]
    assert 'six' not in CONTRACTS[family]
    assert '600' not in CONTRACTS[family]
    assert '0.75' not in CONTRACTS[family]


@pytest.mark.parametrize('family', FAMILIES)
def test_every_controlled_world_gold_passes_and_legacy_omits_extra_behavior(family):
    states = protected_states(family, 2890002)
    sample = witness(family)
    for mask in range(64):
        source = oracle_source(family, mask)
        namespace = {}
        exec(source, namespace)
        missing_extra = False
        for state in states:
            actual = namespace['solve'](copy.deepcopy(state))
            assert canonical(actual) == canonical(expected(family, state, mask))
            missing_extra |= canonical(legacy_expected(family, state, mask)) != canonical(actual)
        assert missing_extra, (family, mask)
        bounded = run_program(source, sample)
        assert bounded['ok'], bounded['error']
        assert canonical(bounded['result']) == canonical(expected(family, sample, mask))
        legacy = run_program(legacy_oracle_source(family, mask), sample)
        assert legacy['ok']
        assert canonical(legacy['result']) != canonical(bounded['result'])


def test_drag_lift_is_output_only_and_never_accumulates_or_changes_hit_testing():
    f = 'drag_hysteresis'
    state = state_after(f, [dict(kind='down', x=105, y=255),
        dict(kind='move', x=120, y=265), dict(kind='move', x=125, y=269)])
    for mask in range(64):
        current, original = expected(f, state, mask), legacy_expected(f, state, mask)
        assert current['active_id'] == original['active_id']
        assert current['layout'] == original['layout']
        active = current['active_id']
        for key, position in current['positions'].items():
            difference = [a-b for a, b in zip(position, original['positions'][key])]
            assert difference == ([0, -8, 0, 0] if key == active else [0, 0, 0, 0])
        released = transition(f, state, dict(kind='up'))
        assert expected(f, released, mask) == legacy_expected(f, released, mask)


def test_chart_clamps_current_and_historical_projection():
    f = 'linked_brush'
    state = state_after(f, [dict(kind='set_transform', zoom=2, pan_x=0, pan_y=0),
        dict(kind='brush', x0=130, y0=540, x1=145, y1=570, mode='replace'),
        dict(kind='set_transform', zoom=.5, pan_x=0, pan_y=0)])
    equivalent = copy.deepcopy(state)
    equivalent['events'][0]['zoom'] = 1.5
    equivalent['events'][2]['zoom'] = .75
    for mask in range(64):
        # Legacy at the effective clamped values is equivalent for every
        # historical/current binding choice and selection geometry.
        assert expected(f, state, mask) == legacy_expected(f, equivalent, mask)


def test_physics_clamps_both_signs_after_gravity_each_tick():
    f = 'contact_dynamics'
    for vy, gravity, expected_vy in [(590, 1000, 600), (-590, -1000, -600)]:
        state = state_after(f, [dict(kind='set_body', x=0, y=100, vx=0, vy=vy),
            dict(kind='set_physics', gravity=gravity, dt=.05), dict(kind='advance', ticks=2)])
        for mask in range(64):
            result = expected(f, state, mask)
            assert result['vy'] == expected_vy
            assert result['y'] == pytest.approx(100+expected_vy*.1)


@pytest.mark.parametrize('family', FAMILIES)
def test_extra_behavior_has_private_witness_without_seventh_bit_credit(family):
    env = VisualBatch4Env(family=family, reward_mode='train')
    env.reset(seed=2890003, options={'world_seed': world(family, 0)})
    for event in witness(family)['events']:
        obs, *_ = env.step({'type': 'interact', 'event': event})
    report = env.research_report()
    assert report['novel_witnesses'][-1]['extra_behavior_visible']
    assert report['information_bits'] == 0
    assert report['reward_ledger'][-1]['components']['information'] == 0
    assert not report['mutants_killed']
    assert 'extra_behavior_visible' not in obs['context']
    assert 'novel_witnesses' not in obs['context']
    # Both panels still use unchanged 1200x700 RGB drawing and visible footers.
    assert obs['image'].shape == (700, 1200, 3)
    assert not np.array_equal(obs['image'][:, :600], obs['image'][:, 600:])


@pytest.mark.parametrize('family', FAMILIES)
def test_grade_uses_new_truth_and_correct_gold_has_no_verification_quota(family):
    env = VisualBatch4Env(family=family, max_steps=2)
    env.reset(seed=2890004, options={'world_seed': 12})
    legacy = env._grade(legacy_oracle_source(family, env._mask))
    assert not legacy['success']
    env.step({'type': 'edit', 'source': oracle_source(family, env._mask)})
    _, reward, terminated, truncated, _ = env.step({'type': 'submit'})
    assert terminated and not truncated and env.final_result['success']
    assert env.final_result['total'] == {'drag_hysteresis': 27, 'linked_brush': 20, 'contact_dynamics': 25}[family]
    assert reward == pytest.approx(9.975)
    assert not env.research_report()['latest_revision_tests']


def test_batch4_rejects_proposal_curricula_and_default_timeout():
    with pytest.raises(ValueError):
        VisualBatch4Env(candidate_mode='audit_incomplete')
    env = VisualBatch4Env()
    env.reset(seed=2890005, options={'world_seed': 1})
    for _ in range(16):
        _, _, terminated, truncated, _ = env.step({'type': 'observe'})
    assert truncated and not terminated and env.steps == 16
    assert env.research_report()['reward_ledger'][-1]['components']['budget_exhaustion'] == -2


def test_public_contract_is_verbatim_supplied_redacted_novel_and_only_new_id_registered():
    import gymnasium as gym
    import visual_repair_lab
    assert visual_repair_lab.__version__ == '7.0.0'
    encoded = json.dumps(CONTRACTS, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == 'e0427f4e1b19a19ac8325d4687ef5c25f22a26e89bbebf58ed41a49ac3a65915'
    assert 'VisualTrainingRepair-v0' in gym.registry
    assert 'VisualRepairBatch4-v0' not in gym.registry
    assert 'VisualRepairReward-v1' not in gym.registry
    assert 'VisualCausalRepair-v0' not in gym.registry
    assert 'VisualDiscoveryRepair-v0' not in gym.registry
