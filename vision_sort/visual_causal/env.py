"""Sequence-dependent visual experiments, executable repairs and private audits."""
import copy
import hashlib
import json
import math
import secrets
import string
from collections import Counter

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from ..visual_discovery.runtime import run_program
from .tasks import (CONTROLS, CONTRACTS, FACTORS, FAMILIES, canonical, expected,
                    initial_state, starter_source, panel, protected_states, transition, valid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class VisualCausalEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, family='drag_hysteresis', max_steps=40, image_access=True,
                 objective='repair', reward_mode='evaluation', gamma=.97, render_mode='rgb_array'):
        super().__init__()
        if family not in FAMILIES or objective not in ('repair', 'investigate', 'test_design'):
            raise ValueError('Unknown task')
        if type(max_steps) is not int or not 1 <= max_steps <= 200 or type(image_access) is not bool:
            raise ValueError('Invalid budget/image access')
        if reward_mode not in ('train', 'evaluation') or not 0 < gamma <= 1 or render_mode != 'rgb_array':
            raise ValueError('Invalid reward/render mode')
        self.family, self.max_steps, self.image_access = family, max_steps, image_access
        self.objective, self.reward_mode, self.gamma = objective, reward_mode, gamma
        self.render_mode = render_mode
        self.action_space = spaces.Text(20000, min_length=1, charset=string.printable)
        self.observation_space = spaces.Dict(image=spaces.Box(0, 255, (700, 1200, 3), np.uint8),
                                            context=spaces.Text(100000, min_length=1, charset=string.printable))
        self._nonce = secrets.randbits(128)
        self.starter = starter_source(family)
        namespace = {}
        exec(self.starter, namespace)
        self._starter_solve = namespace['solve']
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        private = options.get('world_seed', secrets.randbits(128) if seed is None else int(digest([self._nonce, seed]), 16) % 2**128)
        if type(private) is not int or not 0 <= private < 2**128:
            raise ValueError('Invalid private seed')
        self._world_seed = private
        self._mask = int(digest([private, self.family]), 16) % 64
        self.episode_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.state = initial_state(self.family, self.episode_seed)
        self.source = self.active_source = self.starter
        self.source_revision = self.active_revision = 0
        self.steps = 0
        self.terminated = self.truncated = False
        self.reward = self.task_return = self.shaping_return = 0.0
        self.final_result = None
        self._posterior = set(range(64))
        self._kills, self._final_kills, self._credited = set(), set(), set()
        self._metrics, self._claims, self._final_tests = [], [], []
        self._seen_states = set()
        self.history = []
        self.crop_box = None
        self.message = 'Repair the candidate to match the approved visual behavior. Choose your own experiments and tests.'
        self._execute()
        self._refresh()
        self._measure('reset')
        self._initial_entropy = self.entropy
        self._metrics = []
        self._ready = True
        return self._observation(), self._info()

    @property
    def entropy(self):
        return math.log2(len(self._posterior))

    def _execute(self):
        if self.active_source == self.starter:
            value = self._starter_solve(copy.deepcopy(self.state))
            result = {'ok': True, 'result': value, 'error': ''}
        else:
            result = run_program(self.active_source, self.state)
        ok = result['ok'] and valid(self.family, self.state, result['result'])
        self.output = result['result'] if ok else None
        self.public_result = dict(executed=result['ok'], schema_valid=bool(ok),
                                  error=result['error'] if not result['ok'] else '' if ok else 'Output schema is invalid.')

    def _refresh(self):
        self._reference = panel(self.family, self.state, expected(self.family, self.state, self._mask), 'Approved reference')
        self._candidate = panel(self.family, self.state, self.output, 'Current candidate')
        self._image = np.concatenate([self._reference, self._candidate], axis=1)

    def _context(self):
        return dict(environment='VisualCausalRepair-v0', family=self.family, objective=self.objective,
                    prompt=CONTRACTS[self.family] + ' All actions cost budget. Saving does not activate code; run activates it. '
                    'Before submission, execution feedback reports schema only, never reference correctness. '
                    'Every graded reference output is visible in the image, with numeric grading at one decimal. '
                    'No verification sequence or coverage quota is required. Candidate solve(state) must replay the entire public event history.',
                    interactions=CONTROLS[self.family], operations=['observe', 'crop', 'interact', 'edit', 'run', 'submit'],
                    source=self.source, source_revision=self.source_revision, active_revision=self.active_revision,
                    scene=self.state, public_result=self.public_result, message=self.message,
                    steps=self.steps, budget=self.max_steps, remaining=self.max_steps-self.steps,
                    terminated=self.terminated, truncated=self.truncated, final_result=self.final_result,
                    crop_box=self.crop_box, image_access=self.image_access)

    def _observation(self):
        return {'context': json.dumps(self._context(), ensure_ascii=True),
                'image': self._image.copy() if self.image_access else np.zeros_like(self._image)}

    def _info(self):
        return dict(tool_steps=self.steps, reward=self.reward, terminated=self.terminated, truncated=self.truncated)

    def render(self):
        return self._observation()['image']

    def _measure(self, operation):
        before, kills = self.entropy, set()
        expected_gain = 0.0
        full = self.image_access and operation in ('reset', 'interact', 'run', 'observe')
        truth = canonical(expected(self.family, self.state, self._mask))
        if full:
            key = digest(self.state)
            if key not in self._seen_states:
                values = {i: digest(canonical(expected(self.family, self.state, i))) for i in self._posterior}
                groups = Counter(values.values())
                expected_gain = before - sum(n/len(values)*math.log2(n) for n in groups.values())
                observed = digest(truth)
                self._posterior = {i for i, h in values.items() if h == observed}
                assert self._mask in self._posterior
                self._seen_states.add(key)
            kills = {FACTORS[self.family][i] for i in range(6)
                     if canonical(expected(self.family, self.state, self._mask ^ (1 << i))) != truth}
        new = kills - self._kills
        self._kills |= kills
        final_test = full and self.source_revision > 0 and self.active_revision == self.source_revision and self.public_result['schema_valid']
        matches = self.output is not None and canonical(self.output) == truth
        if final_test:
            self._final_kills |= kills
            self._credited |= kills
            self._final_tests.append(dict(step=self.steps, state_sha256=digest(self.state), rendered_match=matches))
        self._metrics.append(dict(step=self.steps, operation=operation, information_gain=before-self.entropy,
                                 expected_information_gain=expected_gain, remaining_worlds=len(self._posterior),
                                 new_mutants_killed=sorted(new), mutants_killed_here=sorted(kills),
                                 final_revision_test=final_test, visible_match=matches if full else None,
                                 source_revision=self.source_revision, active_revision=self.active_revision))

    def record_claims(self, claims):
        """Evaluator-only scoring of optional declared visible facts, no feedback."""
        truth = canonical(expected(self.family, self.state, self._mask))
        for claim in claims:
            row = dict(step=self.steps, path=claim['path'], value_json=claim['value_json'], observable=self.image_access)
            try:
                value = truth
                for key in claim['path'].strip('/').split('/'):
                    value = value[int(key)] if isinstance(value, list) else value[key]
                predicted = json.loads(claim['value_json'])
                row.update(valid_path=True, correct=canonical(predicted) == value if self.image_access else None)
            except (KeyError, IndexError, TypeError, ValueError):
                row.update(valid_path=False, correct=None)
            self._claims.append(row)

    def _potential(self):
        info = (self._initial_entropy - self.entropy) / max(1, self._initial_entropy)
        selected = self._kills if self.objective == 'test_design' else self._credited
        power = len(selected & set(FACTORS[self.family][:4])) / 4
        return info if self.objective == 'investigate' else power if self.objective == 'test_design' else (info+power)/2

    def _grade(self, source):
        rows = []
        for i, state in enumerate(protected_states(self.family, self.episode_seed + 100000)):
            result = run_program(source, state)
            ok = result['ok'] and valid(self.family, state, result['result'])
            agrees = ok and canonical(result['result']) == canonical(expected(self.family, state, self._mask))
            rows.append(dict(case=i, schema_valid=bool(ok), passed=bool(agrees)))
        return dict(success=all(r['passed'] for r in rows), passed=sum(r['passed'] for r in rows),
                    total=len(rows), checks=rows, scored_fields='All output fields drawn at one-decimal precision')

    def step(self, action):
        if not self._ready or self.terminated or self.truncated:
            raise gym.error.ResetNeeded('Reset before stepping')
        before = self._potential()
        self.steps += 1
        task_reward, operation = -.025, 'invalid'
        try:
            if isinstance(action, str):
                if len(action) > 20000:
                    raise ValueError('Action too large')
                action = json.loads(action)
            if type(action) is not dict:
                raise ValueError('Action must be an object')
            operation = action.get('type')
            if operation == 'interact':
                self.state = transition(self.family, self.state, action.get('event'))
                self._execute()
                self.crop_box = None
                self._refresh()
                self.message = 'Interaction applied to active source.'
            elif operation == 'observe':
                self.crop_box = None
                self._refresh()
                self.message = 'Full approved reference and candidate.'
            elif operation == 'crop':
                box = action.get('box')
                if type(box) is not list or len(box) != 4 or any(type(v) not in (int, float) or not np.isfinite(v) for v in box):
                    raise ValueError('Invalid crop')
                x, y, w, h = map(int, box)
                if x < 0 or y < 0 or w < 8 or h < 8 or x+w > 1200 or y+h > 700:
                    raise ValueError('Invalid crop bounds')
                self.crop_box = [x, y, w, h]
                full = np.concatenate([self._reference, self._candidate], axis=1)
                self._image = np.asarray(Image.fromarray(full).crop((x, y, x+w, y+h)).resize((1200, 700)))
                self.message = 'Enlarged already delivered pixels.'
            elif operation == 'edit':
                source = action.get('source')
                if type(source) is not str or not source.strip() or len(source.encode()) > 16000:
                    raise ValueError('Invalid source')
                if source != self.source:
                    self.source, self.source_revision = source, self.source_revision+1
                    self._final_kills, self._final_tests = set(), []
                self.message = 'Source saved; run activates it.'
            elif operation == 'run':
                self.active_source, self.active_revision = self.source, self.source_revision
                self._execute()
                self.crop_box = None
                self._refresh()
                self.message = 'Schema-valid execution; inspect the result.' if self.public_result['schema_valid'] else self.public_result['error']
            elif operation == 'submit':
                if self.objective == 'repair':
                    self.final_result = self._grade(self.source)
                    task_reward += 10 if self.final_result['success'] else -2
                else:
                    score = ((self._initial_entropy-self.entropy)/max(1, self._initial_entropy)
                             if self.objective == 'investigate' else len(self._kills & set(FACTORS[self.family][:4]))/4)
                    self.final_result = dict(kind='probe_curriculum', score=score, success=score >= 1, repair_success_measured=False)
                    task_reward += 10*score
                self.terminated = True
                self.message = 'Submitted; protected evaluation complete.'
            else:
                raise ValueError('Unknown operation')
        except (ValueError, TypeError, KeyError, OverflowError):
            self.message = 'Invalid action. Check public control fields and bounds.'
            operation, task_reward = 'invalid', task_reward-.1
        self.truncated = self.steps >= self.max_steps and not self.terminated
        self._measure(operation)
        after = 0.0 if self.terminated else self._potential()
        shaping = self.gamma*after-before if self.reward_mode == 'train' else 0.0
        self.reward = task_reward+shaping
        self.task_return += task_reward
        self.shaping_return += shaping
        self._metrics[-1].update(task_reward=task_reward, shaping_reward=shaping)
        self.history.append(copy.deepcopy(action))
        return self._observation(), self.reward, self.terminated, self.truncated, self._info()

    def research_report(self):
        correct_claims = [r for r in self._claims if r.get('correct') is True]
        wrong_claims = [r for r in self._claims if r.get('correct') is False]
        return dict(family=self.family, objective=self.objective, information_bits=self._initial_entropy-self.entropy,
                    initial_entropy_bits=self._initial_entropy, remaining_worlds=len(self._posterior),
                    mutants_killed=sorted(self._kills), rewarded_mutants_killed=len(self._kills & set(FACTORS[self.family][:4])),
                    reward_held_out_mutants_killed=len(self._kills & set(FACTORS[self.family][4:])),
                    latest_revision_test_power=sorted(self._final_kills), latest_revision_tests=self._final_tests,
                    claims=self._claims, correct_visible_claims=len(correct_claims), incorrect_visible_claims=len(wrong_claims),
                    steps=self._metrics, submitted=self.terminated, truncated=self.truncated,
                    task_return=self.task_return, shaping_return=self.shaping_return,
                    final_result=self.final_result, final_source_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
                    notes=['Information is ideal-observer disambiguation over six binary mechanisms, not model understanding.',
                           'Mutants flip one mechanism; kills measure potential test power, not noticed bugs.',
                           'Seeing errors require an explicit incorrect visible claim; absent claims are unmeasured.'])
