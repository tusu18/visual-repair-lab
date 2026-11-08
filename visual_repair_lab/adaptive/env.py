"""V6 adaptive world, retaining v5 reward prerequisites and actor contracts."""
import json
import gymnasium as gym
from vision_sort.visual_training.env import VisualTrainingEnv
from vision_sort.visual_training import tasks
from vision_sort.visual_training.mutants import compile_trusted
from vision_sort.visual_causal.env import digest
from visual_repair_lab.v5.env import GeneratedRepairEnv
from visual_repair_lab.v5.compare import equivalent
from .dsl import validate, seal
from .compiler import compile_source
from .certify import batch
from .probes import grading_states
from .reference import reference


class AdaptiveRepairEnv(GeneratedRepairEnv):
    def __init__(self,spec,reward_mode='evaluation',evidence='image'):
        self.adaptive_spec=validate(spec)
        family=spec['family']
        scaffold={'drag_hysteresis':{'canvas_clamp':False,'wide_gain':1.},
                  'linked_brush':{'pan_limit':20.,'column_a':False},
                  'contact_dynamics':{'dt_max':.04,'gravity_limit':600.}}[family]
        super().__init__(dict(id='backend',family=family,split='train',level='composed',
            mask=spec['world']['mask'],scene_seed=3310000,rules=scaffold),reward_mode=reward_mode,evidence=evidence)
        self.max_steps=spec['budget']
        self.gold_source=compile_source(spec);self._gold=compile_trusted(self.gold_source)

    def reset(self,*,seed=None,options=None):
        spec=self.adaptive_spec
        VisualTrainingEnv.reset(self,seed=spec['scene_seed'],options={'world_seed':spec['scene_seed']})
        if spec['starting_point']=='baseline':
            source=tasks.oracle_source(self.family,spec['world']['mask'],'train')
        elif spec['starting_point']=='proposal':
            source=compile_source(spec,disabled=[len(spec['rules'])-1])
        else:
            source=tasks.starter_source(self.family)
        self.source=self.active_source=self.starter=source
        self.source_revision=self.active_revision=0
        self._starter_solve=compile_trusted(source)
        self._mutants={f'rule_{i}':compile_trusted(compile_source(spec,disabled=[i])) for i in range(len(spec['rules']))}
        self._heldout_mutants={};self._mutant_credit=set()
        self._execute();self._refresh()
        final={digest(s) for s in grading_states(spec,'final')}
        self._progress_cases=[s for s in grading_states(spec,'progress') if digest(s) not in final]
        self._behavior_fingerprints={}
        self._behavior_outputs={}
        passed=self._case_passes(source)
        self._baseline_behavior=self._behavior_fingerprints[digest(source)]
        self._progress_cache={digest(source):passed};self._last_progress_passes=passed.copy()
        self._progress_high=len(passed)/len(self._progress_cases)
        self._metrics=[];self.execution_ledger=[]
        if seed is not None:
            # A task manifest fixes the world. The Gym seed controls only the
            # episode RNG; changing it cannot reveal/resample private settings.
            gym.Env.reset(self,seed=seed)
        return self._observation(),self._info()

    def _grading_truth(self,state=None,mask=None):
        spec=self.adaptive_spec if mask is None else dict(self.adaptive_spec,world={'mask':mask})
        return reference(spec,self.state if state is None else state)

    def _meaningful_change(self,source):
        if not self.public_result.get('schema_valid'):return False
        key=digest(source)
        if key not in self._behavior_fingerprints:self._case_passes(source)
        if self._behavior_fingerprints[key]==self._baseline_behavior:return False
        return not equivalent(self._behavior_outputs[key],self._behavior_outputs[digest(self.starter)])

    def _context(self):
        context=super()._context()
        context['environment']='VisualAdaptiveRepair-v0'
        context['level']='adaptive'
        context['prompt']+=' Candidate code replays all public event history. The reference may contain additional conditional behavior.'
        return context

    def _case_passes(self,source):
        results=batch(source,self._progress_cases)
        if not hasattr(self,'_behavior_fingerprints'):self._behavior_fingerprints={}
        if not hasattr(self,'_behavior_outputs'):self._behavior_outputs={}
        values=[{'status':r['error_code'],'output':r['result'] if r['ok'] and tasks.valid(self.family,s,r['result']) else None} for s,r in zip(self._progress_cases,results)]
        self._behavior_outputs[digest(source)]=values
        self._behavior_fingerprints[digest(source)]=seal(values)
        self._progress_status[digest(source)]=[r['error_code'] for r in results]
        return {i for i,(s,r) in enumerate(zip(self._progress_cases,results))
                if r['ok'] and tasks.valid(self.family,s,r['result']) and equivalent(r['result'],self._grading_truth(s))}

    def _grade(self,source):
        states=grading_states(self.adaptive_spec,'final');results=batch(source,states);rows=[]
        for i,(s,r) in enumerate(zip(states,results)):
            valid=bool(r['ok'] and tasks.valid(self.family,s,r['result']))
            rows.append(dict(case=i,schema_valid=valid,passed=valid and equivalent(r['result'],self._grading_truth(s)),execution_status=r['error_code']))
        active=self.source_revision==self.active_revision
        return dict(success=all(r['passed'] for r in rows),passed=sum(r['passed'] for r in rows),total=len(rows),checks=rows,
            final_revision_activated=active,chosen_verification_tests=len(self._final_tests) if active else 0,
            chosen_tests_all_match=bool(active and self._final_tests and all(r['rendered_match'] for r in self._final_tests)),
            verification_mutants_exposed=sorted(self._final_kills) if active else [])

    def research_report(self):
        r=super().research_report();r.update(version='6.0.0',environment='VisualAdaptiveRepair-v0',
            spec_sha256=seal(self.adaptive_spec),notes=['Adaptive task delivery and sampling; no VLM weight updates.'])
        return r
