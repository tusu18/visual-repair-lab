"""Numeric-observation catalogue-search proxy. No private target lookup.

The complete frozen TRAIN program corpus is a declared privileged prior. The
policy sees only public contexts and that same corpus for every attempt. It has
no private test programs, task ID, certificate, final cases or target settings.
This is not a VLM and cannot establish visual perception or training gains.
"""
import copy
import json
import random
from vision_sort.visual_training.mutants import compile_trusted
from vision_sort.visual_training import tasks
from .certify import visible_signature
from .probes import experiments


class CatalogueAgent:
    def __init__(self,programs,seed):
        self.programs=[(source,compile_trusted(source)) for source in programs]
        self.remaining=list(range(len(self.programs)));self.rng=random.Random(seed)
        self.plan=[];self.phase='search';self.chosen=None

    def act(self,context):
        family=context['family'];state=context['scene'];budget=context['remaining']
        target=context['displayed_values']['reference']
        signature=visible_signature(target)
        survivors=[]
        for index in self.remaining:
            try:
                if visible_signature(self.programs[index][1](copy.deepcopy(state)))==signature:survivors.append(index)
            except (KeyError,TypeError,ValueError,ZeroDivisionError):pass
        self.remaining=survivors
        if self.phase=='edit':self.phase='run';return {'type':'run'}
        if self.phase=='run':
            self.phase='verified';return {'type':'interact','event':{'kind':'reset_scene'}} if budget>1 else {'type':'submit'}
        if self.phase=='verified':return {'type':'submit'}
        if self.plan:
            return {'type':'interact','event':self.plan.pop(0)}
        if len(self.remaining)<=1 or budget<=5:
            choices=self.remaining or list(range(len(self.programs)))
            self.chosen=self.rng.choice(choices);self.phase='edit'
            return {'type':'edit','source':self.programs[self.chosen][0]}
        choices=[sequence for sequence in experiments(family) if len(sequence)+1<=budget-4]
        if not choices:
            self.phase='edit';return {'type':'edit','source':self.programs[self.rng.choice(self.remaining)][0]}
        # A finite, randomized experiment search makes success depend on actual
        # observations and remaining budget; no success coin is sampled.
        sample=self.rng.sample(choices,min(3,len(choices)))
        scored=[]
        for sequence in sample:
            candidate=copy.deepcopy(state)
            try:
                candidate=tasks.transition(family,candidate,{'kind':'reset_scene'})
                for event in sequence:candidate=tasks.transition(family,candidate,event)
                predictions={visible_signature(self.programs[i][1](copy.deepcopy(candidate))) for i in self.remaining}
                scored.append((len(predictions)/(len(sequence)+1),self.rng.random(),sequence))
            except (KeyError,TypeError,ValueError):pass
        sequence=max(scored,key=lambda x:x[:2])[2] if scored else self.rng.choice(choices)
        self.plan=[{'kind':'reset_scene'}]+copy.deepcopy(sequence)
        return {'type':'interact','event':self.plan.pop(0)}


def run_attempt(job):
    from .env import AdaptiveRepairEnv
    spec,programs,seed,attempt_id=job
    env=AdaptiveRepairEnv(spec,evidence='numeric_text');observation,_=env.reset()
    agent=CatalogueAgent(programs,seed);trace=[]
    while not (env.terminated or env.truncated):
        context=json.loads(observation['context']);action=agent.act(context)
        observation,reward,*_=env.step(action)
        trace.append({'action':action,'reward':reward,'context':json.loads(observation['context'])})
    final=env.final_result
    return {'attempt_id':attempt_id,'status':'completed','success':bool(final and final['success']),
            'passed':final['passed'] if final else None,'total':final['total'] if final else None,
            'return':env.task_return,'actions':env.steps,'trace':trace,'source_sha256':env.research_report()['final_source_sha256']}
