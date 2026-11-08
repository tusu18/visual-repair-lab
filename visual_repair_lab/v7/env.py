"""Keep absolute correctness separate from reward for repairing the initial code."""
from visual_repair_lab.adaptive.env import AdaptiveRepairEnv
from visual_repair_lab.adaptive.private_cv import DetectionRepairEnv
from visual_repair_lab.v5.compare import equivalent
from vision_sort.visual_causal.env import digest


def leaf_agreement(actual, expected):
    """Public-schema fields, raw tolerance; this never changes the final grader."""
    if isinstance(expected,dict):
        scores=[leaf_agreement(actual.get(k) if isinstance(actual,dict) else None,v) for k,v in expected.items()]
    elif isinstance(expected,list):
        scores=[leaf_agreement(actual[i] if isinstance(actual,list) and i<len(actual) else None,v) for i,v in enumerate(expected)]
    else:
        return float(equivalent(actual,expected))
    return sum(scores)/len(scores) if scores else float(equivalent(actual,expected))


def terminal_reward(start, final, total):
    """No payment for inherited correctness; full *improvement* earns ten."""
    if final <= start:
        return -2.0 * (start-final) / total
    improvement = (final-start)/(total-start)
    return 8.0*improvement + (2.0 if final == total else 0.0)


class RelativeReward:
    def reset(self, **kwargs):
        super().reset(**kwargs)
        initial = self._grade(self.source)
        self._initial_passed = initial['passed']
        self._initial_total = initial['total']
        if hasattr(self,'_behavior_outputs'):
            self._field_scores={}
            self._field_last=self._field_high=self._field_score(self.source)
        return self._observation(), self._info() if hasattr(self, '_info') else {}

    def _field_score(self,source):
        key=digest(source)
        if key not in self._field_scores:
            if key not in self._behavior_outputs:self._case_passes(source)
            rows=self._behavior_outputs[key]
            self._field_scores[key]=sum(leaf_agreement(r['output'],self._grading_truth(s))
                if r['output'] is not None else 0.0 for s,r in zip(self._progress_cases,rows))/len(rows)
        return self._field_scores[key]

    def _context(self):
        c = super()._context()
        c['environment'] = 'VisualRepairDelivery-v7'
        c['prompt'] = c['prompt'].replace('Submit gives partial correctness credit.',
            'Submit pays only for improvement over the initial candidate; unchanged behavior earns zero terminal reward.')
        c.setdefault('reward_contract', {}).update(
            partial_submit='8 * max(0, final_passed - initial_passed) / (total - initial_passed); +2 for improved full repair',
            unchanged_submit_terminal=0.0,
            regression_submit='-2 * max(0, initial_passed - final_passed) / total',
            action=-.025, unactivated_submission=-.25,
            progress='Train-only field agreement improvement on the disjoint progress pool, cap 0.75',
            test_power_max_per_action=.1,
            already_correct_start='No terminal payment for preserving an already correct start',
            baseline_grade_is_private_until_submit=True)
        return c

    def step(self, action):
        progress_paid=getattr(self,'_positive',{}).get('repair_progress',0.0)
        result = super().step(action)
        row=self.reward_ledger[-1]
        c=row['components']
        if hasattr(self,'_field_scores') and row['operation'] in ('edit','run','submit'):
            score=self._field_score(self.source)
            self._positive['repair_progress']=progress_paid
            c['repair_progress']=self._bounded('repair_progress',.75*max(0.0,score-self._field_high))
            c['regression']=-.75*max(0.0,self._field_last-score)
            self._field_high=max(self._field_high,score);self._field_last=score
        if c.get('test_power',0.0)>.1:
            self._positive['test_power']-=c['test_power']-.1
            c['test_power']=.1
        if self.terminated:
            c['terminal'] = terminal_reward(self._initial_passed,
                self.final_result['passed'], self._initial_total)
        base_keys = ('action', 'invalid', 'terminal', 'budget_exhaustion', 'unactivated_submission')
        base = sum(c.get(k, 0.0) for k in base_keys)
        process = sum(v for k, v in c.items() if k not in base_keys)
        applied = process if self.mode == 'train' else 0.0
        row.update(base_reward=base, process_reward=process,
            applied_process_reward=applied, reward=base+applied)
        self.task_return = sum(r.get('base_reward', r['reward']) for r in self.reward_ledger)
        self.process_return = sum(r.get('applied_process_reward', 0.0) for r in self.reward_ledger)
        self.reward = base+applied
        row['cumulative_reward'] = self.task_return+self.process_return
        return self._observation(), self.reward, self.terminated, self.truncated, result[-1]

    def research_report(self):
        r = super().research_report()
        r.update(version='7.0.0', environment='VisualRepairDelivery-v7',
            initial_grade={'passed':self._initial_passed, 'total':self._initial_total},
            reward_version='relative-start-v1')
        return r


class RepairEnv(RelativeReward, AdaptiveRepairEnv):
    pass


class DetectionEnv(RelativeReward, DetectionRepairEnv):
    pass


def make_env(spec, **kwargs):
    return (DetectionEnv if spec['family']=='detection_overlay' else RepairEnv)(spec, **kwargs)
