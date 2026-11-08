"""Adversarial public-contract tests for the new experiment reward version.

All fixtures are development worlds. Oracle sources are trusted test fixtures,
never actor inputs or measured model repairs.
"""
import copy
import json

import numpy as np
import pytest

from vision_sort.visual_causal.env import digest
from vision_sort.visual_experiments.env import LIMITS, VisualExperimentEnv
from vision_sort.visual_experiments.tasks import (
    FACTORS,
    canonical,
    expected,
    oracle_source,
    protected_states,
    transition,
)


def private_seed_for(family, mask):
    for world_seed in range(100_000):
        if int(digest([world_seed, family, 'v3']), 16) % 512 == mask:
            return world_seed
    raise AssertionError('Development world search exhausted')


def episode(family='linked_brush', mask=511, mode='train', images=True, max_steps=100):
    env = VisualExperimentEnv(family=family, reward_mode=mode, image_access=images,
                              split='development', max_steps=max_steps)
    env.reset(seed=3010000, options={'world_seed': private_seed_for(family, mask)})
    return env


def interact(env, **event):
    return env.step({'type': 'interact', 'event': event})


def activate_oracle(env):
    source = oracle_source(env.family, env._mask, env.split)
    env.step({'type': 'edit', 'source': source})
    env.step({'type': 'run'})
    assert env.public_result['schema_valid']
    return source


def components(env):
    return env.reward_ledger[-1]['components']


def report(env, path, reference, candidate, observation_id=None):
    return env.step({'type': 'report', 'event': {
        'observation_id': observation_id or env.frame_id, 'path': path,
        'reference_json': json.dumps(reference), 'candidate_json': json.dumps(candidate),
    }})


@pytest.mark.parametrize('family,events', [
    ('drag_hysteresis', [
        {'kind': 'down', 'x': 105, 'y': 255}, {'kind': 'set_zoom', 'zoom': 2},
        {'kind': 'move', 'x': 122, 'y': 255}, {'kind': 'up'},
    ]),
    ('linked_brush', [
        {'kind': 'set_transform', 'zoom': .5, 'pan_x': 0, 'pan_y': 0},
        {'kind': 'set_transform', 'zoom': 2, 'pan_x': 0, 'pan_y': 0},
        {'kind': 'set_transform', 'zoom': 1.2, 'pan_x': 20, 'pan_y': -10},
    ]),
    ('contact_dynamics', [
        {'kind': 'set_body', 'x': 0, 'y': 220, 'vx': 0, 'vy': 1000},
        {'kind': 'set_physics', 'gravity': 900, 'dt': .05}, {'kind': 'advance', 'ticks': 1},
        {'kind': 'set_body', 'x': 0, 'y': 220, 'vx': 0, 'vy': -1000},
        {'kind': 'set_physics', 'gravity': -900, 'dt': .05}, {'kind': 'advance', 'ticks': 1},
    ]),
])
def test_new_behavior_axes_are_observable_and_earn_experiment_reward(family, events):
    env = episode(family)
    before = env.entropy
    for event in events:
        env.step({'type': 'interact', 'event': event})
    assert set(FACTORS[family][6:]).issubset(env._kills)
    assert env.entropy < before
    assert sum(row['components']['information'] for row in env.reward_ledger) > 0


def test_extreme_chart_probes_add_information_after_ordinary_view_probes():
    env = episode()
    interact(env, kind='set_view', context='A')
    interact(env, kind='set_view', context='B')
    entropy_before = env.entropy
    interact(env, kind='set_transform', zoom=.5, pan_x=0, pan_y=0)
    assert components(env)['information'] > 0
    interact(env, kind='set_transform', zoom=2, pan_x=0, pan_y=0)
    assert components(env)['information'] > 0
    assert env.entropy < entropy_before


def test_repeated_views_and_crops_cannot_farm_information():
    env = episode()
    event = {'kind': 'set_transform', 'zoom': .5, 'pan_x': 20, 'pan_y': -10}
    env.step({'type': 'interact', 'event': event})
    assert components(env)['information'] > 0
    for action in [{'type': 'observe'}, {'type': 'crop', 'box': [0, 0, 600, 700]},
                   {'type': 'observe'}, {'type': 'interact', 'event': event}]:
        env.step(action)
        assert components(env)['information'] == 0


def test_only_interactions_verify_latest_revision_and_source_cycles_do_not_repay():
    env = episode()
    source = activate_oracle(env)
    assert components(env)['test_power'] == 0
    env.step({'type': 'observe'})
    assert components(env)['test_power'] == 0
    event = {'kind': 'set_transform', 'zoom': .5, 'pan_x': 20, 'pan_y': -10}
    env.step({'type': 'interact', 'event': event})
    assert components(env)['test_power'] > 0
    assert env._final_tests
    paid = env._positive['test_power']
    env.step({'type': 'edit', 'source': source + '\n'})
    assert not env._final_tests and not env._final_kills
    env.step({'type': 'interact', 'event': event})
    assert not env._final_tests
    assert components(env)['test_power'] == 0
    env.step({'type': 'run'})
    env.step({'type': 'interact', 'event': event})
    assert env._final_tests
    assert components(env)['test_power'] == 0
    assert env._positive['test_power'] == paid
    env.step({'type': 'edit', 'source': source})
    env.step({'type': 'run'})
    env.step({'type': 'interact', 'event': event})
    assert components(env)['test_power'] == 0
    assert env._positive['test_power'] == paid


def test_progress_high_water_does_not_pay_fix_regress_fix_cycle():
    env = episode()
    source = activate_oracle(env)
    first_progress = components(env)['repair_progress']
    assert first_progress > 0
    env.step({'type': 'edit', 'source': env.starter})
    env.step({'type': 'run'})
    assert components(env)['regression'] < 0
    assert components(env)['repair_progress'] == 0
    env.step({'type': 'edit', 'source': source})
    env.step({'type': 'run'})
    assert components(env)['repair_progress'] == 0
    assert env._positive['repair_progress'] == pytest.approx(first_progress)
    assert sum(env._positive.values()) <= sum(LIMITS.values())


def test_unactivated_saved_source_can_pass_but_earns_no_verification_claim():
    env = episode()
    env.step({'type': 'edit', 'source': oracle_source(env.family, env._mask, env.split)})
    _, reward, terminated, truncated, _ = env.step({'type': 'submit'})
    assert terminated and not truncated
    assert reward == pytest.approx(9.975)
    assert env.final_result['success']
    assert not env.final_result['final_revision_activated']
    assert env.final_result['chosen_verification_tests'] == 0
    assert not env.final_result['chosen_tests_all_match']
    assert not env.final_result['verification_mutants_exposed']


def test_reports_bind_current_observation_and_deduplicate_mechanism_across_fields():
    env = episode()
    interact(env, kind='set_transform', zoom=.5, pan_x=0, pan_y=0)
    old_frame = env.frame_id
    truth = canonical(env._truth())['projected']['S1'][0]
    actual = canonical(env.output)['projected']['S1'][0]
    report(env, '/projected/S1/0/0', truth[0], actual[0])
    assert components(env)['report'] > 0
    report(env, '/projected/S1/0/1', truth[1], actual[1])
    assert components(env)['report'] == 0
    report(env, '/projected/S1/0/0', truth[0], actual[0], observation_id=old_frame)
    assert components(env)['invalid'] == -.1
    assert components(env)['report'] == 0


def test_self_introduced_defect_has_no_original_defect_credit():
    env = episode(mask=0)
    # At the initial chart state, the unedited starter exactly matches truth.
    assert canonical(env._truth()) == canonical(env.output)
    candidate = copy.deepcopy(env.output)
    candidate['projected']['S1'][0][0] += 20
    source = 'def solve(s):\n    return ' + repr(candidate) + '\n'
    env.step({'type': 'edit', 'source': source})
    env.step({'type': 'run'})
    truth = canonical(env._truth())['projected']['S1'][0][0]
    actual = canonical(env.output)['projected']['S1'][0][0]
    report(env, '/projected/S1/0/0', truth, actual)
    assert components(env)['report'] == 0
    assert not env._report_credit


def test_forecast_is_prospective_atomic_and_duplicate_axes_are_not_repaid():
    env = episode()
    event = {'kind': 'set_transform', 'zoom': .5, 'pan_x': 0, 'pan_y': 0}
    next_state = transition(env.family, env.state, event)
    predicted = canonical(expected(env.family, next_state, env._mask, env.split))['projected']['S1'][0][0]
    action = {'type': 'interact', 'event': event, 'prediction': {
        'observation_id': env.frame_id, 'path': '/projected/S1/0/0',
        'value_json': json.dumps(predicted),
    }}
    env.step(action)
    assert components(env)['prediction'] > 0
    first_payment = env._positive['prediction']
    assert env.statement_ledger[-1]['source_observation_id'] == action['prediction']['observation_id']
    assert env.statement_ledger[-1]['target_observation_id'] == env.frame_id
    env.step(action)  # Binding is now stale, even though the numeric value matches.
    assert components(env)['invalid'] == -.1
    assert components(env)['prediction'] == 0
    interact(env, kind='reset_scene')
    action['prediction']['observation_id'] = env.frame_id
    env.step(action)
    assert components(env)['prediction'] == 0
    assert env._positive['prediction'] == first_payment


def test_wrong_prospective_forecast_is_penalized_without_correctness_text():
    env = episode()
    event = {'kind': 'set_transform', 'zoom': .5, 'pan_x': 0, 'pan_y': 0}
    obs, *_ = env.step({'type': 'interact', 'event': event, 'prediction': {
        'observation_id': env.frame_id, 'path': '/projected/S1/0/0', 'value_json': '-999',
    }})
    assert components(env)['prediction'] == -.1
    assert json.loads(obs['context'])['message'] == 'Interaction applied to active source.'


def test_private_worlds_have_identical_policy_text_and_distinct_visuals():
    left, right = episode(mask=0, mode='evaluation'), episode(mask=64, mode='evaluation')
    for event in [{'kind': 'set_transform', 'zoom': .5, 'pan_x': 0, 'pan_y': 0},
                  {'kind': 'set_view', 'context': 'A'}]:
        obs_left, *_ = left.step({'type': 'interact', 'event': event})
        obs_right, *_ = right.step({'type': 'interact', 'event': event})
        assert obs_left['context'] == obs_right['context']
        assert not np.array_equal(obs_left['image'], obs_right['image'])
    public = json.loads(obs_left['context'])
    for private_key in ('world_seed', '_mask', 'parameter_split', 'remaining_worlds',
                        'mutants_killed', 'reward_ledger', 'statement_ledger', 'information_bits'):
        assert private_key not in public


def test_train_and_evaluation_share_observations_but_apply_different_scalar_rewards():
    train, evaluation = episode(mode='train'), episode(mode='evaluation')
    for event in [{'kind': 'set_transform', 'zoom': .5, 'pan_x': 20, 'pan_y': -10},
                  {'kind': 'set_transform', 'zoom': 2, 'pan_x': 20, 'pan_y': -10}]:
        a = {'type': 'interact', 'event': event}
        obs_train, r_train, *_ = train.step(a)
        obs_eval, r_eval, *_ = evaluation.step(a)
        assert obs_train['context'] == obs_eval['context']
        assert np.array_equal(obs_train['image'], obs_eval['image'])
        ledger = train.reward_ledger[-1]
        assert r_train == pytest.approx(sum(ledger['components'].values()))
        assert r_eval == pytest.approx(ledger['base_reward'])
        assert r_train > r_eval
        assert evaluation.reward_ledger[-1]['applied_process_reward'] == 0


def test_source_only_has_no_image_dependent_reward_or_nonzero_image():
    env = episode(images=False)
    obs, *_ = interact(env, kind='set_transform', zoom=.5, pan_x=20, pan_y=-10)
    assert not np.any(obs['image'])
    assert all(components(env)[name] == 0 for name in ('information', 'test_power', 'prediction', 'report'))
    report(env, '/projected/S1/0/0', 0, 0)
    assert components(env)['report'] == 0
    assert components(env)['invalid'] == 0


@pytest.mark.parametrize('family', FACTORS)
def test_progress_and_final_histories_are_disjoint_and_every_axis_has_a_witness(family):
    progress = protected_states(family, 3010000, 'development', pool='progress')
    final = protected_states(family, 3010000, 'development', pool='final')
    assert set(map(digest, progress)).isdisjoint(map(digest, final))
    witnesses = set()
    # A fixed all-enabled world makes conditional factors such as snap-grid live.
    for state in final:
        truth = canonical(expected(family, state, 511, 'development'))
        for i in range(9):
            if truth != canonical(expected(family, state, 511 ^ (1 << i), 'development')):
                witnesses.add(i)
    assert witnesses == set(range(9))


def test_optional_claim_audit_uses_new_reference_instead_of_legacy_fixed_lift():
    env = episode('drag_hysteresis', mask=511)
    interact(env, kind='down', x=105, y=255)
    y = canonical(env._truth())['positions']['A'][1]
    assert y == 208  # Development high-lift value 12, rather than Batch4's 8.
    env.record_claims([
        {'path': '/positions/A/1', 'value_json': json.dumps(y)},
        {'path': '/positions/A/1', 'value_json': '212'},
    ])
    audit = env.research_report()
    assert [claim['correct'] for claim in audit['claims']] == [True, False]


def test_fabricated_reports_cannot_receive_credit_and_private_paths_do_not_leak():
    env = episode()
    interact(env, kind='set_transform', zoom=.5, pan_x=0, pan_y=0)
    obs, *_ = report(env, '/projected/S1/0/0', -999, -999)
    assert components(env)['report'] == -.1
    acknowledgement = json.loads(obs['context'])['message']
    obs, *_ = report(env, '/selected_ids/100', -999, -999)
    assert components(env)['report'] == 0
    assert json.loads(obs['context'])['message'] == acknowledgement
    assert not env._report_credit


def test_unsubmitted_training_timeout_cannot_avoid_failure_penalty():
    env = episode(max_steps=1)
    _, reward, terminated, truncated, _ = env.step({'type': 'observe'})
    assert not terminated and truncated
    assert components(env)['budget_exhaustion'] == -2
    assert components(env)['terminal'] == 0
    assert reward == pytest.approx(-2.025)


def test_running_and_observing_without_a_new_test_cannot_earn_final_verification():
    env = episode()
    activate_oracle(env)
    env.step({'type': 'observe'})
    env.step({'type': 'submit'})
    assert env.final_result['success']
    assert components(env)['verification'] == 0


@pytest.mark.parametrize('reactivate_after_edit', [False, True])
def test_earlier_revision_tests_do_not_pay_final_verification(reactivate_after_edit):
    env = episode()
    source = activate_oracle(env)
    interact(env, kind='set_transform', zoom=.5, pan_x=20, pan_y=-10)
    assert env._final_tests and env._final_tests[-1]['rendered_match']
    env.step({'type': 'edit', 'source': source + '\n'})
    if reactivate_after_edit:
        env.step({'type': 'run'})
    env.step({'type': 'submit'})
    assert env.final_result['success']
    assert components(env)['verification'] == 0


@pytest.mark.parametrize('mode', ['train', 'evaluation'])
def test_matching_final_test_earns_fractional_bonus_only_on_submission(mode):
    env = episode(mode=mode)
    activate_oracle(env)
    # In overview, this exposes only the lower-bound and zoom-scaled-pan axes.
    interact(env, kind='set_transform', zoom=.5, pan_x=20, pan_y=-10)
    assert env._final_tests[-1]['rendered_match']
    assert components(env)['verification'] == 0
    assert env._positive['verification'] == 0
    _, reward, terminated, truncated, _ = env.step({'type': 'submit'})
    assert terminated and not truncated
    assert env.final_result['success']
    assert components(env)['verification'] == pytest.approx(.2 * 2 / 7)
    expected_reward = 9.975 + (.2 * 2 / 7 if mode == 'train' else 0)
    assert reward == pytest.approx(expected_reward)


def test_observed_mismatch_exposes_test_power_without_earning_matching_verification():
    env = episode()
    wrong_lower_bound = oracle_source(env.family, env._mask ^ (1 << 6), env.split)
    env.step({'type': 'edit', 'source': wrong_lower_bound})
    env.step({'type': 'run'})
    interact(env, kind='set_transform', zoom=.5, pan_x=20, pan_y=-10)
    assert components(env)['test_power'] > 0
    assert not env._final_tests[-1]['rendered_match']
    env.step({'type': 'submit'})
    assert not env.final_result['success']
    assert components(env)['verification'] == 0


def test_final_bonus_still_encourages_retest_after_prior_exposure_credit_is_spent():
    from gymnasium.error import ResetNeeded

    env = episode()
    wrong_lower_bound = oracle_source(env.family, env._mask ^ (1 << 6), env.split)
    env.step({'type': 'edit', 'source': wrong_lower_bound})
    env.step({'type': 'run'})
    event = {'kind': 'set_transform', 'zoom': .5, 'pan_x': 20, 'pan_y': -10}
    env.step({'type': 'interact', 'event': event})
    assert components(env)['test_power'] > 0
    assert not env._final_tests[-1]['rendered_match']
    activate_oracle(env)
    for _ in range(3):
        env.step({'type': 'interact', 'event': event})
        assert env._final_tests[-1]['rendered_match']
        assert components(env)['test_power'] == 0
        assert components(env)['verification'] == 0
    env.step({'type': 'submit'})
    assert env.final_result['success']
    assert components(env)['verification'] == pytest.approx(.2 * 2 / 7)
    assert env._positive['verification'] == pytest.approx(.2 * 2 / 7)
    with pytest.raises(ResetNeeded):
        env.step({'type': 'submit'})


def test_source_only_correct_matching_runs_get_no_final_visual_verification_bonus():
    env = episode(images=False)
    activate_oracle(env)
    interact(env, kind='set_transform', zoom=.5, pan_x=20, pan_y=-10)
    env.step({'type': 'submit'})
    assert env.final_result['success']
    assert components(env)['verification'] == 0
