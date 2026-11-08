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
from .runtime import checked_run as run_program, checked_batch, language_contract, EvaluationInfrastructureError
from .mutants import mutant_sources, compile_trusted, exposure
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

LIMITS = dict(information=0., test_power=.5, verification=.25, prediction=.25,
              report=0., comparison=.25, repair_progress=.75)
PROFILES = {
    'terminal': set(),
    'progress': {'repair_progress', 'regression'},
    'tests': {'repair_progress', 'regression', 'test_power', 'verification'},
    'claims': {'repair_progress', 'regression', 'test_power', 'verification', 'comparison', 'prediction'},
    'full': {'repair_progress', 'regression', 'test_power', 'verification', 'comparison', 'prediction'},
}



class VisualTrainingEnv(VisualCausalEnv):
    """Nine controlled dimensions; correctness and verification kept separate."""

    def __init__(self, family='drag_hysteresis', max_steps=48, image_access=True,
                 reward_mode='evaluation', split='train', curriculum='composed',
                 reward_profile='progress', candidate_mode='mixed', implementation='split',
                 appearance='standard',
                 **kwargs):
        if reward_mode not in ('train', 'evaluation'):
            raise ValueError('Unknown reward mode')
        if split not in ('train', 'development', 'validation', 'test'):
            raise ValueError('Unknown parameter split')
        if curriculum not in ('foundation', 'composed'):
            raise ValueError('Unknown curriculum')
        if reward_profile not in PROFILES:
            raise ValueError('Unknown reward profile')
        if candidate_mode not in ('starter', 'mixed', 'audit_correct', 'audit_incomplete'):
            raise ValueError('Unknown candidate mode')
        if implementation not in ('split', 'direct', 'renamed', 'dispatch'):
            raise ValueError('Unknown implementation variant')
        if appearance not in ('standard','alternate'):
            raise ValueError('Unknown appearance; footer removal is not validated')
        self.appearance = appearance
        self.profile, self.candidate_mode, self.implementation = reward_profile, candidate_mode, implementation
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
        raw = int(digest([private, self.family, 'v4']), 16)
        masks = list(range(512))
        if self.split in ('train', 'validation', 'test') and self.curriculum == 'composed':
            parity = 0 if self.split == 'train' else 1
            masks = [m for m in masks if m.bit_count() % 2 == parity]
        self._mask = masks[raw % len(masks)]
        # Foundation fixes the six older factors; the three additional ones still
        # vary. The public curriculum discloses this easier distribution.
        if self.curriculum == 'foundation':
            self._mask &= 448
        self.episode_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.state = initial_state(self.family, self.episode_seed)
        from .tasks import starter_source, oracle_source
        from .variants import implementation_variant
        train_sources, heldout_sources = mutant_sources(self.family, self._mask, self.split)
        self._mutants = {k: compile_trusted(v) for k,v in train_sources.items()}
        self._heldout_mutants = {k: compile_trusted(v) for k,v in heldout_sources.items()}
        self._mutant_credit, self._comparison_credit, self._forecast_credit = set(), set(), set()
        self._structural_tests, self.comparison_ledger, self.execution_ledger = [], [], []
        self._progress_status = {}
        self.action_ledger = []
        self._aborted = False
        mode = self.candidate_mode
        if mode == 'mixed':
            mode = ('starter', 'starter', 'audit_correct', 'audit_incomplete')[(raw >> 16) % 4]
        self._candidate_kind = mode
        if mode == 'audit_incomplete':
            probes = protected_states(self.family,self.episode_seed,self.split,pool='progress')
            names = list(train_sources)
            start = (raw >> 24) % len(names)
            ordered = names[start:] + names[:start]
            selected = next((name for name in ordered if any(
                canonical(self._mutants[name](copy.deepcopy(state))) != canonical(self._truth(state))
                for state in probes)), None)
            if selected is None:
                raise RuntimeError('No witnessed incomplete candidate for this world')
        source = (starter_source(self.family) if mode == 'starter' else
                  oracle_source(self.family, self._mask, self.split) if mode == 'audit_correct' else
                  train_sources[selected])
        style = self.implementation
        if style == 'split':
            style = ('direct' if raw % 2 else 'renamed') if self.split in ('train','development') else 'dispatch'
        self._implementation_kind = style
        self.starter = implementation_variant(source, style)
        self._starter_solve = compile_trusted(self.starter)
        self.source = self.active_source = self.starter
        self.source_revision = self.active_revision = self.steps = 0
        self.terminated = self.truncated = False
        self.reward = self.task_return = self.shaping_return = self.process_return = 0.
        self.final_result = None
        self._posterior = set(masks) if self.curriculum == 'composed' else set(range(0, 512, 64))
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
        context.update(environment='VisualTrainingRepair-v0', prompt=CONTRACTS[self.family] + (
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
        context['language_contract'] = language_contract()
        context['appearance'] = self.appearance
        context['reward_profile'] = self.profile
        context['prompt'] += (' The candidate may already be correct; changing it is not required. '
            'Comparisons may be attached to any ordinary action and refer to the BEFORE-action full frame. '
            'Include observation_id, source_revision, path, reference_json, candidate_json and verdict '
            '(match or mismatch). A field match does not claim the entire repair is correct. '
            'Instruction-fuel exhaustion is a candidate error; infrastructure failures abort without a score.')
        context['action_examples']['comparison'] = {'type':'observe', 'comparison': {
            'observation_id':'<current>', 'source_revision':0, 'path':'<visible JSON pointer>',
            'reference_json':'<JSON value>', 'candidate_json':'<JSON value>', 'verdict':'mismatch'}}
        context['reward_contract'].update(positive_process_caps=LIMITS,
            regression_per_lost_fraction=-.75, active_profile=self.profile,
            information_is_diagnostic_only=True)
        if self.curriculum == 'foundation':
            context['prompt'] += (' Foundation curriculum: the older behaviors in the starter are correct; '
                                  'additional visual behaviors may need repair.')
        return context

    def _refresh(self):
        self._reference = panel(self.family, self.state, self._truth(), 'Approved reference')
        self._candidate = panel(self.family, self.state, self.output, 'Current candidate')
        if self.appearance == 'alternate':
            # A channel permutation is invertible: no geometric/label information is erased.
            self._reference = self._reference[:,:, [2,1,0]].copy()
            self._candidate = self._candidate[:,:, [2,1,0]].copy()
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
        final_test = bool(full and operation == 'interact'
                          and self.active_revision == self.source_revision
                          and self.public_result['schema_valid'])
        matches = self.output is not None and canonical(self.output) == truth
        if final_test:
            self._final_kills |= kills
            self._final_tests.append(dict(step=self.steps, state_sha256=digest(self.state),
                source_sha256=digest(self.active_source), rendered_match=matches,
                mutants_killed_here=sorted(kills)))
        structural = exposure(self._mutants, self.state, truth) if full else set()
        heldout = exposure(self._heldout_mutants, self.state, truth) if full else set()
        if final_test:
            self._structural_tests.append(dict(step=self.steps, source_sha256=digest(self.active_source),
                source_revision=self.source_revision, rendered_match=matches,
                rewarded=sorted(structural), heldout=sorted(heldout)))
        self._metrics.append(dict(structural_exposed=sorted(structural),
            heldout_structural_exposed=sorted(heldout), step=self.steps, operation=operation,
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
        results = checked_batch(source,self._progress_cases)
        self._progress_status[digest(source)] = [r['error_code'] for r in results]
        return {index for index,(state,result) in enumerate(zip(self._progress_cases,results))
                if result['ok'] and valid(self.family,state,result['result'])
                and canonical(result['result']) == canonical(self._truth(state))}

    def _grade(self, source):
        rows = []
        states = protected_states(self.family,self.episode_seed,self.split,pool='final')
        results = checked_batch(source,states)
        for index, (state,result) in enumerate(zip(states,results)):
            schema_ok = bool(result['ok'] and valid(self.family, state, result['result']))
            passed = schema_ok and canonical(result['result']) == canonical(self._truth(state))
            rows.append(dict(case=index, schema_valid=schema_ok, passed=bool(passed), execution_status=result['error_code']))
        active = self.active_revision == self.source_revision
        return dict(success=all(r['passed'] for r in rows), passed=sum(r['passed'] for r in rows),
            total=len(rows), checks=rows, final_revision_activated=active,
            chosen_verification_tests=len(self._final_tests) if active else 0,
            chosen_tests_all_match=bool(active and self._final_tests and all(r['rendered_match'] for r in self._final_tests)),
            verification_mutants_exposed=sorted(self._final_kills) if active else [])

    def _bounded(self, name, amount):
        awarded = max(0., min(amount, LIMITS[name]-self._positive[name]))
        self._positive[name] += awarded
        return awarded

    def _execute(self):
        result = run_program(self.active_source, self.state)
        ok = result['ok'] and valid(self.family, self.state, result['result'])
        self.output = result['result'] if ok else None
        self.public_result = dict(executed=result['ok'], schema_valid=bool(ok),
            execution_status=result['error_code'] if not ok else 'ok',
            error=result['error'] if not result['ok'] else '' if ok else 'Output schema is invalid.')
        self.execution_ledger.append(dict(step=self.steps, source_sha256=digest(self.active_source),
            status=result['error_code'], elapsed_ms=result['elapsed_ms']))

    def _comparison(self, statement):
        required = {'observation_id','source_revision','path','reference_json','candidate_json','verdict'}
        if (type(statement) is not dict or set(statement) != required
                or statement['observation_id'] != self.frame_id
                or type(statement['source_revision']) is not int
                or statement['source_revision'] != self.active_revision or self.crop_box is not None
                or statement['verdict'] not in ('match','mismatch')):
            raise ValueError('Invalid comparison binding')
        ref, cand = (canonical(json.loads(statement[k])) for k in ('reference_json','candidate_json'))
        json.dumps([ref,cand],allow_nan=False)
        path = statement['path']
        # Validate paths from the public schema even in the source-only arm.
        actual_ref, actual_cand = field(self._truth(),path), field(self.output,path)
        row = dict(statement, kind='comparison', step=self.steps, correct=None)
        if not self.image_access:
            return row, 0.
        correct = (digest(ref) == digest(actual_ref) and digest(cand) == digest(actual_cand)
                   and (ref == cand) == (statement['verdict'] == 'match'))
        row['correct'] = correct
        if not correct:
            return row, -.1
        original = field(self._starter_solve(copy.deepcopy(self.state)),path) != actual_ref
        witnesses = set()
        if original:
            for name, solve in self._mutants.items():
                if field(solve(copy.deepcopy(self.state)),path) != actual_ref:
                    witnesses.add(name)
        new = witnesses - self._comparison_credit
        self._comparison_credit |= witnesses
        row.update(original_defect=original, new_witnesses=sorted(new))
        return row, self._bounded('comparison', .25*len(new)/len(self._mutants))

    def step(self, action):
        if self._aborted:
            raise gym.error.ResetNeeded('Episode aborted by infrastructure; reset required')
        if not self._ready or self.terminated or self.truncated:
            raise gym.error.ResetNeeded('Reset before stepping')
        if isinstance(action, str):
            try:
                action = json.loads(action) if len(action) <= 20000 else None
            except (ValueError, TypeError, RecursionError):
                action = None
        if type(action) is not dict:
            action = {'type':'invalid'}
        action = copy.deepcopy(action)
        components = dict(action=-.025, invalid=0., terminal=0., budget_exhaustion=0.,
            information=0., test_power=0., comparison=0., prediction=0., verification=0.,
            repair_progress=0., regression=0.)
        original_action = copy.deepcopy(action)
        previous_truth = copy.deepcopy(self._truth())
        previous_frame = self.frame_id
        old_source = self.source
        statement = action.pop('comparison', None)
        prediction = action.pop('prediction', None)
        comparison_row, forecast = None, None
        if action.get('type') == 'report':
            event = action.get('event')
            statement = {}
            if type(event) is dict:
                statement = dict(event)
                statement.setdefault('source_revision', self.active_revision)
                try:
                    statement.setdefault('verdict', 'match' if json.loads(statement['reference_json']) == json.loads(statement['candidate_json']) else 'mismatch')
                except (KeyError, ValueError, TypeError):
                    pass
            action = {'type':'observe'}
        try:
            if statement is not None:
                try:
                    comparison_row, components['comparison'] = self._comparison(statement)
                except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
                    components['invalid'] = -.1
            if prediction is not None:
                try:
                    if (action.get('type') != 'interact' or type(prediction) is not dict
                            or set(prediction) != {'observation_id','path','value_json'}
                            or prediction['observation_id'] != previous_frame or self.crop_box is not None):
                        raise ValueError('Invalid forecast')
                    value = canonical(json.loads(prediction['value_json']))
                    json.dumps(value, allow_nan=False)
                    forecast = (prediction['path'], value)
                except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
                    components['invalid'] = -.1
            # The base handles only public actions. Dynamic dispatch uses v4 truth,
            # execution, grading and metrics; its temporary reward is replaced below.
            super().step(action)
            metric = self._metrics[-1]
            operation = metric['operation']
            if operation == 'invalid':
                components['invalid'] = -.1
            if self.source != old_source:
                self._structural_tests = []
            if metric['final_revision_test'] and self.image_access:
                exposed = set(metric['structural_exposed'])
                new = exposed - self._mutant_credit
                self._mutant_credit |= exposed
                components['test_power'] = self._bounded('test_power', .5*len(new)/len(self._mutants))
            if forecast is not None and operation == 'interact' and self.image_access:
                path, value = forecast
                try:
                    truth = field(self._truth(), path)
                    correct = digest(truth) == digest(value)
                    witnesses = {name for name, solve in self._mutants.items()
                                 if field(solve(copy.deepcopy(self.state)),path) != truth}
                    new = witnesses - self._forecast_credit
                    changed = digest(field(previous_truth,path)) != digest(truth)
                    if correct and changed:
                        self._forecast_credit |= witnesses
                        components['prediction'] = self._bounded('prediction', .25*len(new)/len(self._mutants))
                    elif not correct:
                        components['prediction'] = -.1
                    self.statement_ledger.append(dict(kind='prediction',step=self.steps,correct=correct,
                        source_observation_id=previous_frame,target_observation_id=self.frame_id,
                        path=path,predicted=value,new_witnesses=sorted(new) if correct else []))
                except (ValueError, TypeError, KeyError, IndexError):
                    components['invalid'] = -.1
            if operation == 'run':
                key = digest(self.active_source)
                if key not in self._progress_cache:
                    self._progress_cache[key] = self._case_passes(self.active_source)
                passed = self._progress_cache[key]
                fraction = len(passed)/len(self._progress_cases)
                components['repair_progress'] = self._bounded('repair_progress', .75*max(0.,fraction-self._progress_high))
                components['regression'] = -.75*len(self._last_progress_passes-passed)/len(self._progress_cases)
                self._progress_high = max(self._progress_high,fraction)
                self._last_progress_passes = passed.copy()
            if self.terminated:
                components['terminal'] = 10. if self.final_result['success'] else -2.
                if self.image_access and self.active_revision == self.source_revision:
                    matching = set().union(*(set(r['rewarded']) for r in self._structural_tests if r['rendered_match']))
                    components['verification'] = self._bounded('verification', .25*len(matching)/len(self._mutants))
            if self.truncated and not self.terminated:
                components['budget_exhaustion'] = -2.
        except EvaluationInfrastructureError as exc:
            # No transition/reward may be consumed after a partial evaluator failure.
            self._aborted = True
            self._ready = False
            self.execution_ledger.append(dict(step=self.steps,status=exc.code,episode_credit=False))
            raise
        self.action_ledger.append(original_action)
        if comparison_row is not None:
            self.comparison_ledger.append(comparison_row)
        base = sum(components[k] for k in ('action','invalid','terminal','budget_exhaustion'))
        enabled = PROFILES[self.profile]
        auxiliary = sum(value for name,value in components.items() if name in enabled)
        applied = auxiliary if self.mode == 'train' else 0.
        self.reward = base + applied
        self.task_return = sum(r['base_reward'] for r in self.reward_ledger) + base
        self.process_return += applied
        self.reward_ledger.append(dict(step=self.steps,operation=operation,components=components,
            base_reward=base,process_reward=auxiliary,applied_process_reward=applied,reward=self.reward,
            cumulative_reward=self.task_return+self.process_return,source_revision=self.source_revision,
            active_revision=self.active_revision,reward_profile=self.profile))
        return self._observation(), self.reward, self.terminated, self.truncated, self._info()

    def research_report(self):
        return dict(environment='VisualTrainingRepair-v0',version='4.0.0',family=self.family,
            reward_mode=self.mode,reward_profile=self.profile,parameter_split=self.split,
            curriculum=self.curriculum,candidate_kind=self._candidate_kind,
            implementation_kind=self._implementation_kind,
            information_bits=self._initial_entropy-self.entropy,remaining_worlds=len(self._posterior),
            steps=self._metrics,actions=self.action_ledger,reward_ledger=self.reward_ledger,statement_ledger=self.statement_ledger,
            comparison_ledger=self.comparison_ledger,execution_ledger=self.execution_ledger,
            progress_execution_status=self._progress_status,
            structural_tests=self._structural_tests,positive_process_rewards=self._positive,
            submitted=self.terminated,truncated=self.truncated,aborted=self._aborted,
            final_result=self.final_result,total_return=self.task_return+self.process_return,
            task_return=self.task_return,process_return=self.process_return,
            final_source_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
            notes=['Information gain is diagnostic only; finite structural mutants are not exhaustive.',
                   'Held-out mutation operators are excluded from test-power/claim credit only.',
                   'Parameter and bit-composition splits share mechanism families; inactive factors can alias.',
                   'No training or new model evaluation is claimed.'])
