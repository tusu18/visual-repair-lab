"""Randomized visual system identification and repair, with bounded RL rewards.

Only observations enter actor inputs. research_report() is evaluator-only. The
reward is a supervised auxiliary objective, not policy-invariant shaping.
"""
import copy
import hashlib
import json
import math
import secrets
from collections import Counter

import gymnasium as gym
import numpy as np

from ..visual_causal.env import VisualCausalEnv, digest
from ..visual_discovery.runtime import run_program
from ..visual_reward.env import field
from .tasks import (
    CONTRACTS,
    FACTORS,
    canonical,
    expected,
    initial_state,
    panel,
    protected_states,
    transition,
    valid,
)

LIMITS = dict(information=.8, test_power=.6, verification=.2, prediction=.4, report=.5,
              repair_progress=.8)
REWARDED_FACTORS = (0, 1, 2, 3, 6, 7, 8)


class VisualExperimentEnv(VisualCausalEnv):
    """Nine controlled dimensions; correctness and verification kept separate."""

    def __init__(self, family='drag_hysteresis', max_steps=48, image_access=True,
                 reward_mode='evaluation', split='train', curriculum='composed',
                 **kwargs):
        if reward_mode not in ('train', 'evaluation'):
            raise ValueError('Unknown reward mode')
        if split not in ('train', 'development', 'validation', 'test'):
            raise ValueError('Unknown parameter split')
        if curriculum not in ('foundation', 'composed'):
            raise ValueError('Unknown curriculum')
        self.mode, self.split, self.curriculum = reward_mode, split, curriculum
        super().__init__(family=family, max_steps=max_steps, image_access=image_access,
                         objective='repair', reward_mode='evaluation', **kwargs)

    def _truth(self, state=None, mask=None):
        return expected(self.family, self.state if state is None else state,
                        self._mask if mask is None else mask, self.split)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        private = options.get('world_seed', secrets.randbits(128) if seed is None
                              else int(digest([self._nonce, seed]), 16) % 2**128)
        if type(private) is not int or not 0 <= private < 2**128:
            raise ValueError('Invalid private seed')
        self._world_seed = private
        self._mask = int(digest([private, self.family, 'v3']), 16) % 512
        # Foundation fixes the six older factors; the three additional ones still
        # vary. The public curriculum discloses this easier distribution.
        if self.curriculum == 'foundation':
            self._mask &= 448
        self.episode_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.state = initial_state(self.family, self.episode_seed)
        self.source = self.active_source = self.starter
        self.source_revision = self.active_revision = self.steps = 0
        self.terminated = self.truncated = False
        self.reward = self.task_return = self.shaping_return = self.process_return = 0.
        self.final_result = None
        self._posterior = set(range(512)) if self.curriculum == 'composed' else set(range(0, 512, 64))
        self._metrics, self._claims, self._final_tests, self.history = [], [], [], []
        self._kills, self._final_kills, self._credited, self._seen_states = set(), set(), set(), set()
        self.reward_ledger, self.statement_ledger = [], []
        self._test_credit, self._report_credit, self._prediction_credit = set(), set(), set()
        self._false_credit = set()
        self._positive = {name: 0. for name in LIMITS}
        self.pending_prediction = None
        self.crop_box = None
        self.message = 'Repair the candidate to match the approved reference. Choose experiments and tests.'
        self._execute()
        self._refresh()
        self._measure('reset')
        self._initial_entropy = self.entropy
        self._metrics = []
        final = {digest(s) for s in protected_states(self.family, self.episode_seed, self.split, pool='final')}
        self._progress_cases = [s for s in protected_states(self.family, self.episode_seed, self.split, pool='progress')
                                if digest(s) not in final]
        if not self._progress_cases:
            raise RuntimeError('Disjoint progress cases are required')
        initial_passes = self._case_passes(self.starter)
        self._progress_cache = {digest(self.starter): initial_passes}
        self._last_progress_passes = initial_passes.copy()
        self._progress_high = len(initial_passes) / len(self._progress_cases)
        self._ready = True
        return self._observation(), self._info()

    @property
    def frame_id(self):
        return digest([self.steps, self.state, self.active_source, self.active_revision])[:24]

    def _context(self):
        context = super()._context()
        context.update(environment='VisualExperimentRepair-v0', prompt=CONTRACTS[self.family] + (
            ' All actions cost budget. edit saves code; run activates it. The current image compares '
            'the approved reference (left) with active candidate (right). The footer shows graded values '
            'rounded to one decimal. No required experiment or verification sequence is supplied. '
            'Execution feedback checks syntax and output shape only. Submit grades saved code, even if '
            'not activated; correctness and verification of that revision are reported separately. '
            'Use a function solve with one positional argument; replay its public event history. '
            'Imports and external tools are unavailable. Optional report refers to CURRENT observation_id. '
            'Optional prediction is attached to an interact action and forecasts that exact next event. '
            'Statement acknowledgements contain no correctness feedback.'),
            observation_id=self.frame_id,
            active_source_sha256=hashlib.sha256(self.active_source.encode()).hexdigest(),
            curriculum=self.curriculum,
            action_examples={
                'interact': {'type': 'interact', 'event': {'kind': 'reset_scene'}},
                'edit': {'type': 'edit', 'source': 'def solve(state):\n    return {}\n'},
                'run': {'type': 'run'}, 'submit': {'type': 'submit'},
                'report': {'type': 'report', 'event': {'observation_id': '<current observation_id>',
                    'path': '<visible JSON pointer>', 'reference_json': '<JSON value>', 'candidate_json': '<JSON value>'}},
                'forecast': {'type': 'interact', 'event': {'kind': 'reset_scene'},
                    'prediction': {'observation_id': '<current observation_id>', 'path': '<visible JSON pointer>',
                                   'value_json': '<predicted reference JSON value>'}}},
            reward_contract={'action': -.025, 'invalid': -.1, 'full_repair': 10.,
                'failed_submission_or_timeout': -2., 'positive_process_caps': LIMITS,
                'false_statement': -.1, 'regression_per_lost_fraction': -.8})
        context['operations'] += ['report']
        if self.curriculum == 'foundation':
            context['prompt'] += (' Foundation curriculum: the older behaviors in the starter are correct; '
                                  'additional visual behaviors may need repair.')
        return context

    def _refresh(self):
        self._reference = panel(self.family, self.state, self._truth(), 'Approved reference')
        self._candidate = panel(self.family, self.state, self.output, 'Current candidate')
        self._image = np.concatenate([self._reference, self._candidate], axis=1)

    def _potential(self):
        return 0.  # Base-class shaping is disabled; the explicit ledger is authoritative.

    def _measure(self, operation):
        before = self.entropy
        full = self.image_access and operation in ('reset', 'interact', 'run', 'observe')
        kills, expected_gain = set(), 0.
        truth = canonical(self._truth())
        if full:
            key = digest(self.state)
            if key not in self._seen_states:
                outputs = {m: digest(canonical(self._truth(mask=m))) for m in self._posterior}
                groups = Counter(outputs.values())
                expected_gain = before - sum(n/len(outputs)*math.log2(n) for n in groups.values())
                self._posterior = {m for m, value in outputs.items() if value == digest(truth)}
                if self._mask not in self._posterior:
                    raise RuntimeError('True world excluded')
                self._seen_states.add(key)
            kills = {FACTORS[self.family][i] for i in range(9)
                     if canonical(self._truth(mask=self._mask ^ (1 << i))) != truth}
        self._kills |= kills
        # Only a newly chosen interaction after activating the saved revision is
        # a verification experiment. run/observe cannot recycle old evidence.
        final_test = bool(full and operation == 'interact' and self.source_revision > 0
                          and self.active_revision == self.source_revision
                          and self.public_result['schema_valid'])
        matches = self.output is not None and canonical(self.output) == truth
        if final_test:
            self._final_kills |= kills
            self._final_tests.append(dict(step=self.steps, state_sha256=digest(self.state),
                source_sha256=digest(self.active_source), rendered_match=matches,
                mutants_killed_here=sorted(kills)))
        self._metrics.append(dict(step=self.steps, operation=operation,
            information_gain=before-self.entropy, expected_information_gain=expected_gain,
            remaining_worlds=len(self._posterior), mutants_killed_here=sorted(kills),
            final_revision_test=final_test, visible_match=matches if full else None,
            source_revision=self.source_revision, active_revision=self.active_revision))

    def record_claims(self, claims):
        """Evaluator-only visible assertions, always scored against v3 truth."""
        truth = canonical(self._truth())
        for claim in claims:
            row = dict(step=self.steps, observation_id=self.frame_id, path=claim['path'],
                       value_json=claim['value_json'], observable=self.image_access)
            try:
                value = field(truth, claim['path'])
                predicted = canonical(json.loads(claim['value_json']))
                json.dumps(predicted, allow_nan=False)
                row.update(valid_path=True, correct=(predicted == value) if self.image_access else None)
            except (KeyError, IndexError, TypeError, ValueError, RecursionError):
                row.update(valid_path=False, correct=None)
            self._claims.append(row)

    def _case_passes(self, source):
        passed = set()
        for index, state in enumerate(self._progress_cases):
            if source == self.starter:
                value = self._starter_solve(copy.deepcopy(state))
                ok = valid(self.family, state, value)
            else:
                result = run_program(source, state)
                value, ok = result['result'], result['ok']
                ok = ok and valid(self.family, state, value)
            if ok and canonical(value) == canonical(self._truth(state)):
                passed.add(index)
        return passed

    def _grade(self, source):
        rows = []
        for index, state in enumerate(protected_states(self.family, self.episode_seed, self.split, pool='final')):
            result = run_program(source, state)
            schema_ok = bool(result['ok'] and valid(self.family, state, result['result']))
            passed = schema_ok and canonical(result['result']) == canonical(self._truth(state))
            rows.append(dict(case=index, schema_valid=schema_ok, passed=bool(passed)))
        active = self.active_revision == self.source_revision and self.source_revision > 0
        return dict(success=all(r['passed'] for r in rows), passed=sum(r['passed'] for r in rows),
            total=len(rows), checks=rows, final_revision_activated=active,
            chosen_verification_tests=len(self._final_tests) if active else 0,
            chosen_tests_all_match=bool(active and self._final_tests and all(r['rendered_match'] for r in self._final_tests)),
            verification_mutants_exposed=sorted(self._final_kills) if active else [])

    def _bounded(self, name, amount):
        awarded = max(0., min(amount, LIMITS[name]-self._positive[name]))
        self._positive[name] += awarded
        return awarded

    def _statement(self, action, previous_frame):
        event = action.get('event')
        if type(event) is not dict or set(event) != {'observation_id', 'path', 'reference_json', 'candidate_json'}:
            raise ValueError('Report fields differ')
        if event['observation_id'] != previous_frame or self.crop_box is not None:
            raise ValueError('Report requires current full frame')
        reference = canonical(json.loads(event['reference_json']))
        candidate = canonical(json.loads(event['candidate_json']))
        json.dumps([reference, candidate], allow_nan=False)
        if not self.image_access:
            return {'kind': 'report', 'correct': None}, 0.
        try:
            truth, actual = field(self._truth(), event['path']), field(self.output, event['path'])
        except (ValueError, TypeError, KeyError, IndexError):
            return {'kind': 'report', 'correct': None}, 0.
        correct = reference == truth and candidate == actual
        row = dict(kind='report', correct=correct, observation_id=previous_frame,
                   path=event['path'], reference=reference, candidate=candidate,
                   source_sha256=digest(self.active_source), revision=self.active_revision)
        if not correct:
            key = (digest(self.state), event['path'], digest([reference, candidate]))
            if key not in self._false_credit:
                self._false_credit.add(key)
                return row, -.1
            return row, 0.
        if reference == candidate or self.active_revision != self.source_revision:
            return row, 0.
        try:
            original_defect = field(self._starter_solve(copy.deepcopy(self.state)), event['path']) != truth
        except (ValueError, TypeError, KeyError, IndexError):
            original_defect = False
        axes = set()
        if original_defect:
            for i, name in enumerate(FACTORS[self.family]):
                try:
                    if field(self._truth(mask=self._mask ^ (1 << i)), event['path']) != truth:
                        axes.add(name)
                except (ValueError, TypeError, KeyError, IndexError):
                    pass
        new = axes-self._report_credit
        self._report_credit |= axes
        row.update(original_defect=original_defect, newly_noticed_axes=sorted(new))
        return row, self._bounded('report', len(new)*LIMITS['report']/9)

    def step(self, action):
        if not self._ready or self.terminated or self.truncated:
            raise gym.error.ResetNeeded('Reset before stepping')
        if isinstance(action, str):
            try:
                action = json.loads(action) if len(action) <= 20000 else None
            except (ValueError, TypeError, RecursionError):
                action = None
        if type(action) is not dict:
            action = {'type': 'invalid'}
        action = copy.deepcopy(action)
        previous_frame, before_state = self.frame_id, copy.deepcopy(self.state)
        components = dict(action=-.025, invalid=0., terminal=0., information=0., test_power=0.,
                          prediction=0., report=0., verification=0., repair_progress=0., regression=0., budget_exhaustion=0.)
        statement, forecast, bad_forecast = None, None, False
        if 'prediction' in action:
            try:
                p = action['prediction']
                if (action.get('type') != 'interact' or type(p) is not dict
                        or set(p) != {'observation_id', 'path', 'value_json'}
                        or p['observation_id'] != previous_frame or self.crop_box is not None):
                    raise ValueError('Invalid forecast binding')
                next_state = transition(self.family, self.state, action.get('event'))
                field(self._starter_solve(copy.deepcopy(next_state)), p['path'])
                value = canonical(json.loads(p['value_json']))
                json.dumps(value, allow_nan=False)
                forecast = dict(path=p['path'], value=value)
            except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
                bad_forecast = True
        if action.get('type') == 'report':
            self.steps += 1
            try:
                statement, components['report'] = self._statement(action, previous_frame)
                self.message, operation = 'Statement recorded. No semantic feedback is available.', 'report'
            except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
                self.message, operation = 'Invalid statement. Use the current full observation and public fields.', 'invalid'
                components['invalid'] = -.1
            self.truncated = self.steps >= self.max_steps
            self._measure(operation)
            self.history.append(action)
        else:
            super().step(action)
            operation = self._metrics[-1]['operation']
            if operation == 'invalid' or bad_forecast:
                components['invalid'] = -.1
            if self.terminated:
                components['terminal'] = 10. if self.final_result['success'] else -2.
        metric = self._metrics[-1]
        if self.image_access:
            components['information'] = self._bounded('information', max(0., metric['information_gain'])*.8/9)
            if metric['final_revision_test']:
                reward_ids = {FACTORS[self.family][i] for i in REWARDED_FACTORS}
                exposed = set(metric['mutants_killed_here']) & reward_ids
                new = exposed-self._test_credit
                self._test_credit |= exposed
                components['test_power'] = self._bounded('test_power', len(new)*LIMITS['test_power']/len(REWARDED_FACTORS))
            if forecast is not None and operation == 'interact':
                try:
                    truth = field(self._truth(), forecast['path'])
                    correct = truth == forecast['value']
                    axes = {name for i, name in enumerate(FACTORS[self.family])
                            if field(self._truth(mask=self._mask ^ (1 << i)), forecast['path']) != truth}
                    changed = field(self._truth(before_state), forecast['path']) != truth
                except (ValueError, TypeError, KeyError, IndexError):
                    correct, axes, changed = None, set(), False
                new = axes-self._prediction_credit
                if correct is True and changed and new:
                    self._prediction_credit |= new
                    components['prediction'] = self._bounded('prediction', len(new)*.4/9)
                elif correct is False:
                    components['prediction'] = -.1
                statement = dict(kind='prediction', correct=correct, path=forecast['path'],
                    predicted=forecast['value'], source_observation_id=previous_frame,
                    target_observation_id=self.frame_id, newly_predicted_axes=sorted(new) if correct else [])
        if (self.terminated and self.image_access and self.source_revision > 0
                and self.active_revision == self.source_revision):
            # Once-only terminal bonus for checks of the submitted revision.
            # Edits clear _final_tests, so prior-revision matches cannot earn it.
            matching = set().union(*(set(row['mutants_killed_here'])
                for row in self._final_tests if row['rendered_match']))
            rewarded = {FACTORS[self.family][i] for i in REWARDED_FACTORS}
            components['verification'] = self._bounded('verification',
                LIMITS['verification']*len(matching & rewarded)/len(REWARDED_FACTORS))
        if operation == 'run' and self.active_revision == self.source_revision:
            source_key = digest(self.active_source)
            if source_key not in self._progress_cache:
                self._progress_cache[source_key] = self._case_passes(self.active_source)
            passed = self._progress_cache[source_key]
            fraction = len(passed)/len(self._progress_cases)
            components['repair_progress'] = self._bounded('repair_progress', .8*max(0., fraction-self._progress_high))
            components['regression'] = -.8*len(self._last_progress_passes-passed)/len(self._progress_cases)
            self._progress_high = max(self._progress_high, fraction)
            self._last_progress_passes = passed.copy()
        if statement is not None:
            statement['step'] = self.steps
            self.statement_ledger.append(statement)
        if self.truncated and not self.terminated:
            components['budget_exhaustion'] = -2.
        base = sum(components[k] for k in ('action', 'invalid', 'terminal'))
        process = sum(components.values())-base
        applied = process if self.mode == 'train' else 0.
        self.reward = base+applied
        self.task_return = sum(r['base_reward'] for r in self.reward_ledger)+base
        self.process_return += applied
        row = dict(step=self.steps, operation=operation, components=components, base_reward=base,
                   process_reward=process, applied_process_reward=applied, reward=self.reward,
                   cumulative_reward=self.task_return+self.process_return,
                   source_revision=self.source_revision, active_revision=self.active_revision)
        if not all(math.isfinite(v) for v in components.values()):
            raise RuntimeError('Nonfinite reward')
        self.reward_ledger.append(row)
        return self._observation(), self.reward, self.terminated, self.truncated, self._info()

    def research_report(self):
        return dict(environment='VisualExperimentRepair-v0', version='3.0.0', family=self.family,
            reward_mode=self.mode, parameter_split=self.split, curriculum=self.curriculum,
            information_bits=self._initial_entropy-self.entropy, remaining_worlds=len(self._posterior),
            mutants_killed=sorted(self._kills), latest_revision_test_power=sorted(self._final_kills),
            latest_revision_tests=self._final_tests, steps=self._metrics, reward_ledger=self.reward_ledger,
            statement_ledger=self.statement_ledger, claims=self._claims,
            positive_process_rewards=self._positive,
            original_defect_axes_noticed=sorted(self._report_credit), submitted=self.terminated,
            truncated=self.truncated, final_result=self.final_result,
            task_return=self.task_return, process_return=self.process_return,
            total_return=self.task_return+self.process_return,
            final_source_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
            notes=['Ideal-observer uncertainty is conditional on a nine-axis finite model, not model understanding.',
                   'Inactive parameters may be unidentifiable; full entropy reduction is not a success requirement.',
                   'Test power means potential counterexamples; noticed defects require a correct bound report.',
                   'Factor 4 and 5 mutations are held out of test-power reward only, not every reward term.',
                   'No VLM training or new model baseline is claimed.'])
