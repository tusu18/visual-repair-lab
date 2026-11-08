"""Generated-world evaluator with tolerant grading and revision-bound rewards."""
import copy
import json
import numpy as np
from vision_sort.visual_training.env import VisualTrainingEnv
from vision_sort.visual_training import tasks
from vision_sort.visual_training.runtime import checked_batch
from vision_sort.visual_training.mutants import compile_trusted
from vision_sort.visual_causal.env import digest
from vision_sort.visual_reward.env import field
from .compare import equivalent
from .generator import LEVELS, sources, validate_spec, seal, states_for


class GeneratedRepairEnv(VisualTrainingEnv):
    def __init__(self, spec, reward_mode='evaluation', evidence='image', **kwargs):
        self.spec = validate_spec(spec)
        if evidence not in ('image', 'numeric_text', 'image_no_footer', 'none'):
            raise ValueError('Unknown evidence arm')
        self.evidence = evidence
        self.gold_source, bugs = sources(self.spec)
        self._gold = compile_trusted(self.gold_source)
        self._rule_bugs = {k: compile_trusted(v) for k, v in bugs.items()}
        self._notice_credit = set()
        super().__init__(family=spec['family'], max_steps=LEVELS[spec['level']],
                         reward_mode=reward_mode, split=spec['split'], candidate_mode='starter',
                         reward_profile='claims', image_access=evidence != 'none', **kwargs)

    def reset(self, *, seed=None, options=None):
        self._notice_credit = set()
        observation, info = super().reset(seed=self.spec['scene_seed'], options={'world_seed': self.spec['scene_seed']})
        # Model-generated rules, their omission bugs and nine original-factor
        # counterfactuals share the actual generated reference, never old v4 gold.
        self._mutants = dict(self._rule_bugs)
        for i in range(9):
            other = dict(self.spec, mask=self.spec['mask'] ^ (1 << i), level='composed')
            self._mutants[f'factor_{i}'] = compile_trusted(sources(other)[0])
        self._heldout_mutants = {}
        # These mutants are all reward-visible. No held-out mutation claim.
        self._mutant_credit = set()
        final = {digest(s) for s in states_for(self.spec, 'final')}
        self._progress_cases = [s for s in states_for(self.spec, 'progress') if digest(s) not in final]
        passed = self._case_passes(self.starter)
        self._progress_cache = {digest(self.starter): passed}
        self._last_progress_passes = passed.copy()
        self._progress_high = len(passed)/len(self._progress_cases)
        return self._observation(), info

    def _truth(self, state=None, mask=None):
        return self._gold(copy.deepcopy(self.state if state is None else state))

    def _context(self):
        context = super()._context()
        context['environment'] = 'VisualGeneratedRepair-v0'
        context['prompt'] = (
            'Repair the saved candidate to reproduce the approved reference across interactions. '
            'The reference is the behavioral specification. Choose experiments and tests yourself. '
            'Every environment action costs one budget unit; reconnect and retry of an identical request ID are free. '
            'edit saves code; run activates it; submit grades saved code and ends the episode. '
            'Submit gives partial correctness credit. Unactivated submission costs 0.25 additional reward. '
            'All controls and public histories below are available, but hidden rule values are not supplied. '
            'Use the published Python subset. Optional comparisons refer to the BEFORE-action observation '
            'and active source revision. Numeric claims use displayed one-decimal values. '
            'The initial candidate is not guaranteed correct. No external tools access evaluator files.'
        )
        context['evidence_arm'] = self.evidence
        context['level'] = self.spec['level']
        context['reward_contract'].update(partial_submit='-2 + 8 * passed / total; full pass = 10',
            unactivated_submission=-.25, progress_on=['edit', 'run', 'submit'],
            test_power_requires_changed_active_source=True,
            grading_numeric_tolerance={'absolute': 1e-6, 'relative': 1e-9},
            no_weight_training_claim=True)
        context['action_examples']['comparison']['comparison'].update(path='/active_id', reference_json='"A"', candidate_json='null')
        if self.evidence == 'numeric_text':
            # An explicit ablation: equivalent footer values, not the full image.
            context['displayed_values'] = {'reference': tasks.canonical(self._truth()),
                                          'candidate': tasks.canonical(self.output)}
        return context

    def _observation(self):
        observation = super()._observation()
        if self.evidence in ('numeric_text', 'none'):
            observation['image'] = np.zeros_like(observation['image'])
        return observation

    def _refresh(self):
        super()._refresh()
        if self.evidence == 'image_no_footer':
            # This is an experimental ablation: some contact/velocity fields need
            # multi-frame inference and offscreen values may not be observable.
            self._image[570:, :, :] = (243, 246, 252)

    def _measure(self, operation):
        full = self.image_access and operation in ('reset', 'interact', 'run', 'observe')
        truth = self._truth()
        changed = self.active_source != self.starter and self.active_revision == self.source_revision
        if changed and hasattr(self,'_meaningful_change'):
            changed=self._meaningful_change(self.active_source)
        exposed = {name for name, fn in self._mutants.items()
                   if not equivalent(fn(copy.deepcopy(self.state)), truth)} if full and changed else set()
        match = self.output is not None and equivalent(self.output, truth)
        final_test = bool(full and operation == 'interact' and changed and self.public_result['schema_valid'])
        if final_test:
            self._final_tests.append({'step': self.steps, 'state_sha256': digest(self.state),
                'source_sha256': digest(self.active_source), 'rendered_match': match, 'mutants_killed_here': sorted(exposed)})
            self._final_kills |= exposed
            self._structural_tests.append({'step': self.steps, 'source_sha256': digest(self.active_source),
                'source_revision': self.source_revision, 'rendered_match': match, 'rewarded': sorted(exposed), 'heldout': []})
        self._metrics.append(dict(step=self.steps, operation=operation, structural_exposed=sorted(exposed),
            heldout_structural_exposed=[], information_gain=0., expected_information_gain=0., remaining_worlds=None,
            mutants_killed_here=sorted(exposed), final_revision_test=final_test,
            visible_match=match if full else None, source_revision=self.source_revision, active_revision=self.active_revision))

    def _case_passes(self, source):
        results = checked_batch(source, self._progress_cases)
        self._progress_status[digest(source)] = [r['error_code'] for r in results]
        return {i for i, (s, r) in enumerate(zip(self._progress_cases, results))
                if r['ok'] and tasks.valid(self.family, s, r['result']) and equivalent(r['result'], self._truth(s))}

    def _grade(self, source):
        states = states_for(self.spec, 'final')
        results = checked_batch(source, states)
        rows = []
        for i, (s, r) in enumerate(zip(states, results)):
            valid = bool(r['ok'] and tasks.valid(self.family, s, r['result']))
            rows.append(dict(case=i, schema_valid=valid, passed=valid and equivalent(r['result'], self._truth(s)), execution_status=r['error_code']))
        activated = self.source_revision == self.active_revision
        return dict(success=all(r['passed'] for r in rows), passed=sum(r['passed'] for r in rows), total=len(rows),
            checks=rows, final_revision_activated=activated,
            chosen_verification_tests=len(self._final_tests) if activated else 0,
            chosen_tests_all_match=bool(activated and self._final_tests and all(r['rendered_match'] for r in self._final_tests)),
            verification_mutants_exposed=sorted(self._final_kills) if activated else [])

    def _comparison(self, statement):
        required = {'observation_id','source_revision','path','reference_json','candidate_json','verdict'}
        if (type(statement) is not dict or set(statement) != required
                or statement['observation_id'] != self.frame_id
                or type(statement['source_revision']) is not int
                or statement['source_revision'] != self.active_revision or self.crop_box is not None
                or statement['verdict'] not in ('match','mismatch')):
            raise ValueError('Invalid comparison binding')
        ref, cand = (tasks.canonical(json.loads(statement[k])) for k in ('reference_json','candidate_json'))
        json.dumps([ref,cand],allow_nan=False)
        path = statement['path']
        actual_ref, actual_cand = field(self._truth(),path), field(self.output,path)
        row = dict(statement,kind='comparison',step=self.steps,correct=None)
        if not self.image_access:
            return row, 0.
        # Reports concern displayed values. Grading still uses raw tolerances.
        correct = (ref == tasks.canonical(actual_ref) and cand == tasks.canonical(actual_cand)
                   and (ref == cand) == (statement['verdict'] == 'match'))
        row['correct'] = correct
        if not correct:
            return row, -.1
        defect = not equivalent(actual_cand,actual_ref)
        witnesses = {name for name,solve in self._mutants.items()
                     if not equivalent(field(solve(copy.deepcopy(self.state)),path), actual_ref)} if defect else set()
        new = witnesses - self._comparison_credit
        self._comparison_credit |= witnesses
        row.update(current_defect=defect,new_witnesses=sorted(new))
        return row,self._bounded('comparison',.25*len(new)/len(self._mutants))

    def step(self, action):
        before_source = self.source
        before_truth = copy.deepcopy(self._truth())
        result = super().step(action)
        row = self.reward_ledger[-1]
        c = row['components']
        operation = row['operation']
        # The inherited predictor checks raw fields. Correct it to the public
        # display contract when a finite rounded forecast was otherwise rejected.
        if operation == 'interact' and self.image_access and self.statement_ledger:
            forecast = self.statement_ledger[-1]
            if forecast.get('step') == self.steps and forecast.get('correct') is False:
                truth = field(self._truth(),forecast['path'])
                if tasks.canonical(truth) == forecast['predicted']:
                    forecast['correct'] = True
                    witnesses = {name for name,solve in self._mutants.items()
                        if not equivalent(field(solve(copy.deepcopy(self.state)),forecast['path']),truth)}
                    new = witnesses-self._forecast_credit
                    changed = tasks.canonical(field(before_truth,forecast['path'])) != tasks.canonical(truth)
                    if changed:
                        self._forecast_credit |= witnesses
                    c['prediction'] = self._bounded('prediction',.25*len(new)/len(self._mutants)) if changed else 0.
                    forecast['new_witnesses'] = sorted(new) if changed else []
        if (operation == 'edit' and self.source != before_source) or operation == 'submit':
            key = digest(self.source)
            if key not in self._progress_cache:
                self._progress_cache[key] = self._case_passes(self.source)
            passed = self._progress_cache[key]
            fraction = len(passed)/len(self._progress_cases)
            c['repair_progress'] = self._bounded('repair_progress', .75*max(0., fraction-self._progress_high))
            c['regression'] = -.75*len(self._last_progress_passes-passed)/len(self._progress_cases)
            self._progress_high = max(self._progress_high, fraction)
            self._last_progress_passes = passed.copy()
        if self.terminated:
            f = self.final_result
            c['terminal'] = 10. if f['success'] else -2. + 8.*f['passed']/f['total']
            c['unactivated_submission'] = -.25 if self.active_revision != self.source_revision else 0.
        base_keys = ('action', 'invalid', 'terminal', 'budget_exhaustion', 'unactivated_submission')
        base = sum(c.get(k, 0.) for k in base_keys)
        auxiliary = sum(v for k, v in c.items() if k not in base_keys)
        applied = auxiliary if self.mode == 'train' else 0.
        row.update(base_reward=base, process_reward=auxiliary, applied_process_reward=applied, reward=base+applied)
        self.task_return = sum(r['base_reward'] for r in self.reward_ledger)
        self.process_return = sum(r['applied_process_reward'] for r in self.reward_ledger)
        self.reward = base+applied
        row['cumulative_reward'] = self.task_return+self.process_return
        return self._observation(), self.reward, self.terminated, self.truncated, result[-1]

    def research_report(self):
        report = super().research_report()
        report.update(environment='VisualGeneratedRepair-v0', version='5.0.0',
            spec_sha256=seal(self.spec), evidence_arm=self.evidence,
            information_bits=None, remaining_worlds=None,
            notes=['Finite omission and original-factor mutants; no held-out mutant claim.',
                   'Information gain is disabled for the generated hypothesis space.',
                   'No learned-policy improvement or model training is claimed.'])
        return report
