"""Packaged Batch 4: redacted starter repair with a 16-action default budget."""
from ..visual_causal.env import digest
from ..visual_causal.tasks import canonical, expected, legacy_expected
from ..visual_reward.env import VisualRewardEnv


class VisualBatch4Env(VisualRewardEnv):
    """The unchanged reward design over revised, statically defined truth.

    Candidate audit curricula are deliberately unavailable in this version.
    Auxiliary information metrics still condition on the original six binary
    factors; the deterministic extra behavior is not a seventh posterior bit.
    """

    def __init__(self, family='drag_hysteresis', max_steps=16, image_access=True,
                 reward_mode='evaluation', reward_weights=None,
                 candidate_mode='starter', **kwargs):
        if candidate_mode != 'starter':
            raise ValueError('Batch 4 repairs the original starter only')
        super().__init__(family=family, max_steps=max_steps,
            image_access=image_access, reward_mode=reward_mode,
            reward_weights=reward_weights, candidate_mode='starter', **kwargs)

    def reset(self, *, seed=None, options=None):
        self._novel_witnesses = []
        return super().reset(seed=seed, options=options)

    def _context(self):
        context = super()._context()
        context['environment'] = 'VisualRepairBatch4-v0'
        return context

    def _measure(self, operation):
        super()._measure(operation)
        full = self.image_access and operation in ('reset', 'interact', 'run', 'observe')
        if full:
            current = canonical(expected(self.family, self.state, self._mask))
            original = canonical(legacy_expected(self.family, self.state, self._mask))
            # Evaluator-only diagnostic. It never changes a reward, observation,
            # posterior, mutant definition or grading history.
            self._novel_witnesses.append(dict(
                step=self.steps, operation=operation,
                state_sha256=digest(self.state),
                source_revision=self.source_revision,
                active_revision=self.active_revision,
                extra_behavior_visible=current != original,
                candidate_matches_reference=(self.output is not None
                    and canonical(self.output) == current)))

    def research_report(self):
        report = super().research_report()
        report.update(environment='VisualRepairBatch4-v0',
            task_revision='batch4', novel_witnesses=self._novel_witnesses,
            information_scope='Original six controlled factors, conditional on fixed Batch 4 behavior')
        report['notes'] += [
            'The fixed additional behavior is shared by every controlled world.',
            'Information gain and original mutant kills do not score discovery of that behavior.',
            'Novel-witness diagnostics are evaluator-only and receive no independent reward.',
            'Reward weights, formulas, protected histories, controls and rendering are unchanged.']
        return report
