"""Behavioral checks for reward validity, leakage and credit farming."""
import json
import math

import pytest

from vision_sort.visual_causal.env import digest
from vision_sort.visual_causal.tasks import oracle_source, protected_states
from vision_sort.visual_reward import VisualRewardEnv


def world(family, mask):
    return next(i for i in range(10000) if int(digest([i, family]), 16) % 64 == mask)


def make(mask=1, **kwargs):
    env = VisualRewardEnv(reward_mode='train', **kwargs)
    env.reset(seed=2700001, options={'world_seed': world('drag_hysteresis', mask)})
    return env


def report(env, path, reference, candidate, **extra):
    return {'type': 'report', 'event': dict(frame=env.steps, revision=env.active_revision,
        path=path, reference_json=json.dumps(reference), candidate_json=json.dumps(candidate), **extra)}


def test_true_false_duplicate_and_stale_reports():
    env = make()
    env.step({'type': 'interact', 'event': {'kind': 'resize', 'width': 500}})
    _, reward, _, _, _ = env.step(report(env, '/layout', 'compact', 'wide'))
    assert reward == pytest.approx(.225)
    assert env.research_report()['original_defects_noticed'] == 1
    _, reward, _, _, _ = env.step(report(env, '/layout', 'compact', 'wide'))
    assert reward == pytest.approx(-.025)
    stale = report(env, '/layout', 'compact', 'wide')
    stale['event']['frame'] -= 1
    env.step(stale)
    assert env.reward_ledger[-1]['components']['invalid'] == -.1
    env.step(report(env, '/layout', 'wide', 'compact'))
    assert env.reward_ledger[-1]['components']['report'] == -.1


def test_actor_created_bug_has_no_detection_bonus():
    env = make(mask=0)
    bad = oracle_source('drag_hysteresis', 1)
    env.step({'type': 'edit', 'source': bad})
    env.step({'type': 'run'})
    env.step({'type': 'interact', 'event': {'kind': 'resize', 'width': 500}})
    env.step(report(env, '/layout', 'wide', 'compact'))
    assert env.reward_ledger[-1]['components']['report'] == 0
    assert env.statement_ledger[-1]['correct']
    assert not env.statement_ledger[-1]['original_defect']


def test_progress_is_disjoint_and_cannot_be_farmed_by_reversion():
    env = make()
    final = {digest(s) for s in protected_states(env.family, env.episode_seed+100000)}
    assert final.isdisjoint({digest(s) for s in env._progress_cases})
    good = oracle_source(env.family, env._mask)
    for source in (good, env.starter, good):
        env.step({'type': 'edit', 'source': source})
        env.step({'type': 'run'})
    runs = [r for r in env.reward_ledger if r['operation'] == 'run']
    assert runs[0]['components']['repair_progress'] > 0
    assert runs[1]['components']['regression'] < 0
    assert runs[2]['components']['repair_progress'] == 0


def test_prediction_requires_exact_next_event_and_new_evidence():
    env = make()
    event = {'kind': 'resize', 'width': 500}
    prediction = {'type': 'predict', 'event': dict(event_json=json.dumps(event),
        path='/layout', value_json='"compact"')}
    env.step(prediction)
    env.step({'type': 'interact', 'event': event})
    assert env.reward_ledger[-1]['components']['prediction'] == .05
    env.step(prediction)
    env.step({'type': 'observe'})
    env.step({'type': 'interact', 'event': event})
    assert env.reward_ledger[-1]['components']['prediction'] == 0


def test_eval_records_but_does_not_apply_process_reward_or_leak_scores():
    env = VisualRewardEnv(reward_mode='evaluation')
    env.reset(seed=2700002, options={'world_seed': world(env.family, 1)})
    obs, reward, _, _, info = env.step({'type': 'interact', 'event': {'kind': 'resize', 'width': 500}})
    assert reward == -.025
    assert env.reward_ledger[-1]['process_reward'] > 0
    public = json.loads(obs['context'])
    for name in ('reward_ledger', 'remaining_worlds', 'information_bits', 'original_defects_noticed', 'world_seed'):
        assert name not in public and name not in info


def test_reset_and_time_limit_are_reproducible():
    env = make(max_steps=1)
    options = {'world_seed': world(env.family, 1)}
    obs1, reward1, terminated, truncated, _ = env.step({'type': 'observe'})
    assert truncated and not terminated
    env.reset(seed=2700001, options=options)
    obs2, reward2, _, _, _ = env.step({'type': 'observe'})
    assert obs1['context'] == obs2['context'] and reward1 == reward2


def test_nonfinite_statement_is_rejected_without_poisoning_ledger():
    env = make()
    action = report(env, '/layout', 'compact', 'compact')
    action['event']['reference_json'] = 'NaN'
    env.step(action)
    assert env.reward_ledger[-1]['components']['invalid'] == -.1
    json.dumps(env.research_report(), allow_nan=False)


def test_missing_private_collection_item_has_constant_public_acknowledgement():
    # A list-index probe must not reveal private selection length through text.
    contexts = []
    for mask in (0, 16):
        env = VisualRewardEnv(family='linked_brush')
        env.reset(seed=2700020, options={'world_seed': world('linked_brush', mask)})
        env.step({'type': 'interact', 'event': dict(kind='brush', x0=280, y0=330, x1=320, y1=370, mode='replace')})
        obs, _, _, _, _ = env.step(report(env, '/selected_ids/0', 'S1', 'S1'))
        contexts.append(json.loads(obs['context']))
    assert contexts[0] == contexts[1]


def test_source_only_reports_do_not_query_reference_shape():
    env = VisualRewardEnv(family='linked_brush', image_access=False, reward_mode='train')
    env.reset(seed=2700021, options={'world_seed': 1})
    obs, reward, _, _, _ = env.step(report(env, '/selected_ids/0', 'S1', 'S1'))
    assert json.loads(obs['context'])['message'] == 'Statement recorded. No semantic feedback is available.'
    assert reward == -.025


@pytest.mark.parametrize('family', ['drag_hysteresis', 'linked_brush', 'contact_dynamics'])
def test_proposed_patch_curriculum_has_correct_and_incomplete_candidates(family):
    for mode, passes in [('audit_correct', True), ('audit_incomplete', False)]:
        env = VisualRewardEnv(family=family, candidate_mode=mode)
        obs, _ = env.reset(seed=2700030, options={'world_seed': 21})
        assert 'bits' not in json.loads(obs['context'])['source']
        assert 'audit_correct' not in obs['context'] and 'audit_incomplete' not in obs['context']
        assert env._grade(env.source)['success'] == passes
        before = obs['context']
        again, _ = env.reset(seed=2700030, options={'world_seed': 21})
        assert before == again['context']


@pytest.mark.parametrize('family', ['drag_hysteresis', 'linked_brush', 'contact_dynamics'])
def test_gold_program_passes_without_report_or_verification_quota(family):
    env = VisualRewardEnv(family=family, reward_mode='train')
    env.reset(seed=2700010, options={'world_seed': 12})
    env.step({'type': 'edit', 'source': oracle_source(family, env._mask)})
    _, reward, terminated, truncated, _ = env.step({'type': 'submit'})
    assert terminated and not truncated and env.final_result['success']
    assert reward == pytest.approx(9.975)


def test_audit_information_prior_conditions_on_visible_proposed_code():
    env = VisualRewardEnv(candidate_mode='audit_mixture', reward_mode='train')
    env.reset(seed=2700040, options={'world_seed': 33})
    assert len(env._posterior) == 7 and env._mask in env._posterior
    assert env.entropy == pytest.approx(1+.5*math.log2(6))
    env.step({'type': 'interact', 'event': dict(kind='resize', width=500)})
    env.step({'type': 'observe'})
    assert env.reward_ledger[-1]['components']['information'] == 0


def test_training_timeout_penalty_does_not_change_evaluation_observations():
    kwargs = dict(max_steps=1)
    train = VisualRewardEnv(reward_mode='train', **kwargs)
    evaluate = VisualRewardEnv(reward_mode='evaluation', **kwargs)
    for env in (train, evaluate):
        env.reset(seed=2700041, options={'world_seed': 15})
    a, train_reward, _, _, _ = train.step({'type': 'observe'})
    b, evaluation_reward, _, _, _ = evaluate.step({'type': 'observe'})
    assert a['context'] == b['context']
    assert train_reward == pytest.approx(-2.025)
    assert evaluation_reward == pytest.approx(-.025)
    assert train.reward_ledger[-1]['components']['budget_exhaustion'] == -2
