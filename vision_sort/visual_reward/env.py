"""Auditable, bounded rewards for experiments, noticed defects and repairs.

Private scores never enter observations or public execution messages. RL callers
receive scalar rewards through step; evaluation actors consume observations only.
This deliberately changes the objective, rather than claiming policy invariance.
"""
import copy
from dataclasses import asdict, dataclass
import json
import math

import gymnasium as gym

from ..visual_causal.env import VisualCausalEnv, digest
from ..visual_causal.tasks import FACTORS, canonical, expected, protected_states, transition, valid
from ..visual_discovery.runtime import run_program
from .candidates import candidate_source


@dataclass(frozen=True)
class RewardWeights:
    information_per_bit: float = .1
    mutation_per_factor: float = .15
    verified_defect: float = .25
    correct_prediction: float = .05
    correct_match: float = .025
    false_statement: float = -.1
    repair_progress: float = .8
    regression: float = -.8
    action: float = -.025
    invalid: float = -.1
    full_repair: float = 10.
    failed_submission: float = -2.


def field(value, path):
    if not isinstance(path, str) or not path.startswith('/') or len(path) > 120:
        raise ValueError('A visible field path is required')
    for part in path[1:].split('/'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return canonical(value)


class VisualRewardEnv(VisualCausalEnv):
    """Same three mechanisms, new report/predict actions and separate reward pools."""
    def __init__(self, family='drag_hysteresis', max_steps=48, image_access=True,
                 reward_mode='evaluation', reward_weights=None, candidate_mode='starter', **kwargs):
        self.weights = reward_weights or RewardWeights()
        if candidate_mode not in ('starter', 'audit_mixture', 'audit_correct', 'audit_incomplete'):
            raise ValueError('Unknown proposed-candidate curriculum')
        self.candidate_mode = candidate_mode
        self.v1_mode = reward_mode
        if reward_mode not in ('evaluation', 'train'):
            raise ValueError('Unknown reward mode')
        super().__init__(family=family, max_steps=max_steps, image_access=image_access,
                         objective='repair', reward_mode='evaluation', **kwargs)

    def reset(self, *, seed=None, options=None):
        # Initialize before parent reset dispatches to _context/_info.
        self.reward_ledger, self.statement_ledger = [], []
        self.pending_prediction = None
        self._defect_credit, self._prediction_credit, self._match_credit = set(), set(), set()
        self._test_credit, self._progress_seen = set(), set()
        self._progress_cache = {}
        self._progress_high = 0.
        self._last_progress_passes = set()
        self.process_return = 0.
        self._audit_prior = None
        self._information_credit = 0.
        obs, info = super().reset(seed=seed, options=options)
        if self.candidate_mode != 'starter':
            sample = int(digest([self._world_seed, 'candidate-audit-v1']), 16)
            correct = self.candidate_mode == 'audit_correct' or (self.candidate_mode == 'audit_mixture' and sample % 2 == 0)
            candidate_mask = self._mask if correct else self._mask ^ (1 << ((sample//2) % 6))
            self.starter = candidate_source(self.family, candidate_mask)
            namespace = {}
            exec(self.starter, namespace)
            self._starter_solve = namespace['solve']
            self.source = self.active_source = self.starter
            self._audit_prior = {candidate_mask: .5}
            self._audit_prior.update({candidate_mask ^ (1 << i): 1/12 for i in range(6)})
            self._posterior = set(self._audit_prior)
            self._initial_entropy = self.entropy
            self._execute()
            self._refresh()
            obs = self._observation()
        final_hashes = {digest(s) for s in protected_states(self.family, self.episode_seed + 100000)}
        self._progress_cases = []
        for index, original in enumerate(protected_states(self.family, self.episode_seed + 200000)):
            state = copy.deepcopy(original)
            if self.family == 'drag_hysteresis':
                state['events'].insert(0, dict(kind='set_zoom', zoom=1.1+index%3*.1))
            elif self.family == 'contact_dynamics':
                state['initial_body']['x'] += 1.5+index%3
            if digest(state) not in final_hashes:
                self._progress_cases.append(state)
        if not self._progress_cases:
            raise RuntimeError('A separate progress pool is required')
        self._baseline_progress = self._case_passes(self.starter)
        self._last_progress_passes = self._baseline_progress.copy()
        self._progress_high = len(self._baseline_progress) / len(self._progress_cases)
        self._progress_seen.add(digest(self.starter))
        self._progress_cache[digest(self.starter)] = self._baseline_progress.copy()
        return obs, info

    @property
    def entropy(self):
        prior = getattr(self, '_audit_prior', None)
        if prior is None:
            return super().entropy
        mass = sum(prior[i] for i in self._posterior)
        return -sum((prior[i]/mass)*math.log2(prior[i]/mass) for i in self._posterior)

    def _measure(self, operation):
        prior = getattr(self, '_audit_prior', None)
        gain = None
        if prior is not None:
            gain = 0.
            full = self.image_access and operation in ('reset', 'interact', 'run', 'observe')
            if full and digest(self.state) not in self._seen_states:
                mass = sum(prior[i] for i in self._posterior)
                groups = {}
                for i in self._posterior:
                    key = digest(canonical(expected(self.family, self.state, i)))
                    groups.setdefault(key, []).append(prior[i]/mass)
                conditional = 0.
                for group in groups.values():
                    probability = sum(group)
                    conditional -= probability*sum((p/probability)*math.log2(p/probability) for p in group)
                gain = max(0., self.entropy-conditional)
        super()._measure(operation)
        if gain is not None:
            self._metrics[-1]['expected_information_gain'] = gain

    def _context(self):
        context = super()._context()
        context['environment'] = 'VisualRepairReward-v1'
        # Resolve the known v0 snap-scope wording ambiguity without exposing the
        # episode's choice between snapping and no snapping.
        context['prompt'] = context['prompt'].replace(
            'Up may snap each offset to the nearest20',
            "Up may snap only the active card's offset to the nearest20")
        context['prompt'] += (
            ' Optional report records values from the CURRENT delivered reference and active candidate. '
            'Use event fields frame (current steps), revision (active_revision), path (visible JSON path), '
            'reference_json and candidate_json (JSON-encoded values). Equal values report a match. '
            'Optional predict forecasts the NEXT interaction: event fields event_json (the exact next '
            'control JSON), path, value_json. It does not execute that interaction. '
            'Reports and predictions receive an acknowledgement only, never correctness feedback. '
            'A pending forecast is canceled by any intervening action other than that exact interaction. '
            'Rewards are private training signals; evaluation prompts never include them. '
            'There is no required report, prediction, experiment or verification quota.')
        context['operations'] += ['report', 'predict']
        context['reward_contract'] = asdict(self.weights)
        context['reward_contract_limits'] = dict(information_bits=6, reward_used_mutants=4,
            defect_reports=4, predictions=4, matching_reports=4, progress_bonus=.8,
            held_out_mutants=2, positive_process_cap=3.3)
        if self.candidate_mode != 'starter':
            context['prompt'] += (
                ' This episode audits an externally proposed repair. It may already be correct or '
                'contain an incomplete repair. Choose experiments that could falsify it; keep good '
                'behavior and repair any verified discrepancy. Submit unchanged source if it is correct. '
                'Record any observed counterexample with report so it can be checked reproducibly. '
                'The mixture is half correct proposals; otherwise one of the six controlled settings '
                'is wrong, with equal probability for each setting. '
                'The candidate template exposes its implementation, never whether it matches this reference.')
        return context

    def _case_passes(self, source):
        passed = set()
        for index, state in enumerate(self._progress_cases):
            if source == self.starter:
                value = self._starter_solve(copy.deepcopy(state))
                ok = valid(self.family, state, value)
            else:
                result = run_program(source, state)
                value = result['result'] if result['ok'] else None
                ok = result['ok'] and valid(self.family, state, value)
            if ok and canonical(value) == canonical(expected(self.family, state, self._mask)):
                passed.add(index)
        return passed

    def _statement(self, action):
        event = action.get('event')
        if type(event) is not dict:
            raise ValueError('Statement fields required')
        if action['type'] == 'predict':
            if set(event) != {'event_json', 'path', 'value_json'}:
                raise ValueError('Prediction fields differ')
            predicted_event = json.loads(event['event_json'])
            next_state = transition(self.family, self.state, predicted_event)
            # Validate public field shape using baseline output, not private truth.
            field(self._starter_solve(copy.deepcopy(next_state)), event['path'])
            value = canonical(json.loads(event['value_json']))
            json.dumps(value, allow_nan=False)
            self.pending_prediction = dict(event=predicted_event, path=event['path'], value=value,
                start_state=digest(self.state), step=self.steps)
            return dict(kind='prediction_recorded'), 0.
        if set(event) != {'frame', 'revision', 'path', 'reference_json', 'candidate_json'}:
            raise ValueError('Report fields differ')
        if type(event['frame']) is not int or type(event['revision']) is not int:
            raise ValueError('Integer frame/revision required')
        if event['frame'] != self.steps-1 or event['revision'] != self.active_revision:
            raise ValueError('Report must describe the last delivered frame and active revision')
        reference_value = canonical(json.loads(event['reference_json']))
        candidate_value = canonical(json.loads(event['candidate_json']))
        json.dumps([reference_value, candidate_value], allow_nan=False)
        if not self.image_access:
            return dict(kind='report_unobservable', correct=None), 0.
        # Missing items in a variable-length private selection must not change
        # public acknowledgement or crash an episode. Otherwise report validity
        # would become an unintended text-only reference query.
        try:
            truth = field(expected(self.family, self.state, self._mask), event['path'])
            actual = field(self.output, event['path'])
        except (ValueError, KeyError, TypeError, IndexError):
            return dict(kind='report_unscorable', correct=None), 0.
        correct = reference_value == truth and candidate_value == actual
        mismatch = reference_value != candidate_value
        row = dict(kind='defect_report' if mismatch else 'match_report', correct=correct,
                   path=event['path'], frame=event['frame'], revision=event['revision'],
                   state_sha256=digest(self.state), source_sha256=digest(self.active_source),
                   reference=reference_value, candidate=candidate_value)
        if not correct:
            return row, self.weights.false_statement
        if mismatch:
            # A bug added by the actor earns no detection bonus: the original
            # externally supplied starter must also fail this exact visible field.
            initial = field(self._starter_solve(copy.deepcopy(self.state)), event['path'])
            eligible = initial != truth
            key = event['path']
            novel = key not in self._defect_credit and len(self._defect_credit) < 4
            row.update(original_defect=eligible, novel=novel)
            if eligible and novel:
                self._defect_credit.add(key)
                return row, self.weights.verified_defect
        else:
            key = (digest(self.active_source), digest(self.state))
            if self.source_revision > 0 and self.active_revision == self.source_revision and key not in self._match_credit and len(self._match_credit) < 4:
                self._match_credit.add(key)
                return row, self.weights.correct_match
        return row, 0.

    def step(self, action):
        if not self._ready or self.terminated or self.truncated:
            raise gym.error.ResetNeeded('Reset before stepping')
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except (ValueError, TypeError):
                action = {'type': 'invalid'}
        if type(action) is not dict:
            action = {'type': 'invalid'}
        action = copy.deepcopy(action)
        before_entropy = self.entropy
        pending, self.pending_prediction = self.pending_prediction, None
        components = dict(action=self.weights.action, invalid=0., terminal=0.,
                          information=0., test_power=0., prediction=0., report=0.,
                          repair_progress=0., regression=0., budget_exhaustion=0.)
        statement = None
        if action.get('type') in ('report', 'predict'):
            self.steps += 1
            try:
                statement, components['report'] = self._statement(action)
                self.message = 'Statement recorded. No semantic feedback is available.'
                operation = action['type']
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                self.message = 'Invalid statement. Check public fields, frame and revision.'
                operation = 'invalid'
                components['invalid'] = self.weights.invalid
            self.truncated = self.steps >= self.max_steps
            self._measure(operation)
            self.history.append(action)
        else:
            super().step(action)
            operation = self._metrics[-1]['operation']
            if operation == 'invalid':
                components['invalid'] = self.weights.invalid
            if self.terminated:
                components['terminal'] = self.weights.full_repair if self.final_result['success'] else self.weights.failed_submission
        metric = self._metrics[-1]
        if self.image_access:
            if self._audit_prior is not None:
                gain = metric['expected_information_gain']
                components['information'] = min(max(0., .6-self._information_credit), gain*self.weights.information_per_bit)
                self._information_credit += components['information']
            else:
                components['information'] = max(0., before_entropy-self.entropy)*self.weights.information_per_bit
            if metric['final_revision_test']:
                exposed = set(metric['mutants_killed_here']) & set(FACTORS[self.family][:4])
                new = exposed-self._test_credit
                self._test_credit |= exposed
                components['test_power'] = len(new)*self.weights.mutation_per_factor
            if pending is not None and operation == 'interact' and action.get('event') == pending['event']:
                try:
                    truth = field(expected(self.family, self.state, self._mask), pending['path'])
                    correct = truth == pending['value']
                except (ValueError, KeyError, TypeError, IndexError):
                    truth, correct = None, None
                key = (pending['start_state'], digest(pending['event']), pending['path'])
                novel = key not in self._prediction_credit and len(self._prediction_credit) < 4
                # Correct but uninformative forecasts are measured, not rewarded.
                if novel and correct is not None and metric['information_gain'] > 0:
                    self._prediction_credit.add(key)
                    components['prediction'] = self.weights.correct_prediction if correct else self.weights.false_statement
                statement = dict(kind='prediction_outcome', correct=correct, novel=novel,
                    path=pending['path'], predicted=pending['value'], actual=truth)
        if operation == 'run' and self.active_revision == self.source_revision:
            source_key = digest(self.active_source)
            if source_key not in self._progress_cache:
                self._progress_cache[source_key] = self._case_passes(self.active_source)
            passed = self._progress_cache[source_key]
            fraction = len(passed)/len(self._progress_cases)
            components['repair_progress'] = self.weights.repair_progress*max(0., fraction-self._progress_high)
            components['regression'] = self.weights.regression*len(self._last_progress_passes-passed)/len(self._progress_cases)
            self._progress_high = max(self._progress_high, fraction)
            self._last_progress_passes = passed.copy()
            self._progress_seen.add(source_key)
        if statement is not None:
            statement['step'] = self.steps
            self.statement_ledger.append(statement)
        # Training should not prefer timing out to honestly submitting a failed
        # repair. This auxiliary penalty is never fed to evaluation actors and
        # does not change public observations or evaluation/base reward.
        if self.truncated and not self.terminated:
            components['budget_exhaustion'] = self.weights.failed_submission
        process = sum(v for k, v in components.items() if k not in ('action', 'invalid', 'terminal'))
        applied = process if self.v1_mode == 'train' else 0.
        self.reward = components['action']+components['invalid']+components['terminal']+applied
        self.task_return = sum(r['base_reward'] for r in self.reward_ledger)+components['action']+components['invalid']+components['terminal']
        self.process_return += applied
        self.shaping_return = 0.
        row = dict(step=self.steps, operation=operation, components=components,
            base_reward=components['action']+components['invalid']+components['terminal'],
            process_reward=process, applied_process_reward=applied, reward=self.reward,
            cumulative_reward=self.task_return+self.process_return,
            source_revision=self.source_revision, active_revision=self.active_revision)
        if not all(math.isfinite(v) for v in components.values()):
            raise RuntimeError('Nonfinite reward')
        self.reward_ledger.append(row)
        return self._observation(), self.reward, self.terminated, self.truncated, self._info()

    def research_report(self):
        report = super().research_report()
        report.update(environment='VisualRepairReward-v1', reward_mode=self.v1_mode,
            training_reward_revision='1.1',
            reward_weights=asdict(self.weights), reward_ledger=self.reward_ledger,
            statement_ledger=self.statement_ledger, process_return=self.process_return,
            total_return=self.task_return+self.process_return,
            original_defects_noticed=len(self._defect_credit),
            notes=report['notes']+[
                'Process rewards are bounded auxiliary objectives, not an invariance guarantee.',
                'Progress uses a train-only case pool separate from terminal protected cases.',
                'Reports acknowledge receipt only; no correctness oracle enters model inputs.',
                'No reward is paid for reporting a discrepancy created solely by the actor.'])
        return report
