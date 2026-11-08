"""Finite-catalogue observation decision trees and release acceptance checks."""
import copy
import itertools
import json
import math
from collections import defaultdict
from vision_sort.visual_training import tasks
from vision_sort.visual_training.mutants import compile_trusted
from vision_sort.visual_training.runtime import checked_batch
from visual_repair_lab.v5.compare import equivalent
from .compiler import compile_source
from .dsl import BASE_MASKS, VALUES, seal, validate
from .probes import experiments, grading_states, state_for


def visible_signature(value):
    """Ideal exact footer reader. A declared upper bound, not an OCR model."""
    return seal(tasks.canonical(value))


def palette(spec):
    # The ideal knows a finite configuration palette, never its target label.
    # It must infer the active subset, literals and base world from observations.
    options=[]
    for rule in spec['rules']:
        options.append([None]+[dict(rule,value=v) for v in VALUES[spec['family']][rule['op']]])
    result=[]
    for mask in BASE_MASKS:
        for rules in itertools.product(*options):
            result.append(dict(spec,world={'mask':mask},rules=[r for r in rules if r is not None]))
    return result


def source_allow_empty(spec):
    if spec['rules']:
        return compile_source(spec)
    return tasks.oracle_source(spec['family'],spec['world']['mask'],'train')


def decision_tree(matrix,costs,remaining=None):
    remaining=list(range(len(matrix[0]))) if remaining is None else list(remaining)
    if len(remaining)==1:
        return {'leaf':remaining[0],'cost':0}
    options=[]
    for index,row in enumerate(matrix):
        groups=defaultdict(list)
        for candidate in remaining:groups[row[candidate]].append(candidate)
        if len(groups)>1:
            gain=math.log2(len(remaining))-sum(len(g)/len(remaining)*math.log2(len(g)) for g in groups.values())
            options.append((gain/max(1,costs[index]),-costs[index],-index,groups))
    if not options:
        raise ValueError('unidentifiable_palette')
    _,_,negative_index,groups=max(options,key=lambda r:r[:3]);index=-negative_index
    branches={signature:decision_tree(matrix,costs,group) for signature,group in groups.items()}
    return {'experiment':index,'branches':branches,'cost':costs[index]+max(b['cost'] for b in branches.values())}


def numeric_half_ties(value):
    if isinstance(value,dict):return sum(numeric_half_ties(v) for v in value.values())
    if isinstance(value,list):return sum(numeric_half_ties(v) for v in value)
    if type(value) is float:
        return int(abs(abs(value*10-round(value*10))-.5)<1e-8)
    return 0


def batch(source,states):
    rows=[]
    for offset in range(0,len(states),64):
        rows.extend(checked_batch(source,states[offset:offset+64]))
    return rows


def adjacent(value,direction):
    if isinstance(value,dict):return {k:adjacent(v,direction) for k,v in value.items()}
    if isinstance(value,list):return [adjacent(v,direction) for v in value]
    if type(value) is float:return math.nextafter(value,direction)
    return value


def certify(spec,minimum_checks=2):
    try:validate(spec)
    except ValueError as exc:return {'accepted':False,'reasons':[str(exc)]}
    source=compile_source(spec);solve=compile_trusted(source)
    final=grading_states(spec,'final');progress=grading_states(spec,'progress')
    from .reference import reference
    truth=[reference(spec,s) for s in final]
    neutral=compile_trusted(tasks.oracle_source(spec['family'],spec['world']['mask'],'train'))
    disabled=compile_trusted(compile_source(spec,disabled=range(len(spec['rules']))))
    parity=all(equivalent(neutral(copy.deepcopy(s)),disabled(copy.deepcopy(s))) for s in final+progress)
    reasons=[]
    independent=all(equivalent(solve(copy.deepcopy(s)),reference(spec,s)) for s in final+progress)
    if not independent:reasons.append('independent_reference_mismatch')
    probes=experiments(spec['family'])
    # A reset must erase prior interaction effects, including conditional
    # operation-order choices. Check composed histories, not isolated probes.
    sample=probes if spec['family']=='contact_dynamics' else probes[::max(1,len(probes)//8)]
    tails=[(tail,solve(state_for(spec['family'],spec['scene_seed'],[{'kind':'reset_scene'}]+tail))) for tail in sample]
    reset_ok=all(equivalent(solve(state_for(spec['family'],spec['scene_seed'],prefix+[{'kind':'reset_scene'}]+tail)),expected)
                 for prefix in sample for tail,expected in tails)
    if not reset_ok:reasons.append('reset_history_leak')
    if not parity:reasons.append('neutral_parity')
    execution=batch(source,final)
    runtime_ok=all(r['ok'] and tasks.valid(spec['family'],s,r['result']) and equivalent(r['result'],t)
                   for s,t,r in zip(final,truth,execution))
    if not runtime_ok:reasons.append('oracle_execution')
    kills={}
    for i,rule in enumerate(spec['rules']):
        mutant=compile_trusted(compile_source(spec,disabled=[i]))
        count=sum(not equivalent(mutant(copy.deepcopy(s)),t) for s,t in zip(final,truth))
        kills[str(i)]={'op':rule['op'],'checks_changed':count}
        if count<minimum_checks:reasons.append('rule_effect:'+str(i))
    ties=sum(numeric_half_ties(t) for t in truth)
    # Half-decimal values themselves are legal. They must not create GRADING
    # ties: either neighboring floating-point representation must still pass.
    rounding_stable=all(equivalent(t,adjacent(t,math.inf)) and equivalent(t,adjacent(t,-math.inf)) for t in truth)
    if not rounding_stable:reasons.append('rounding_instability')
    # Descriptions state only observable input/output schemas and controls, so
    # reviewed transformations cannot contradict a fixed behavioral formula.
    contract_ok=all(tasks.valid(spec['family'],s,t) for s,t in zip(final,truth))
    if not contract_ok:reasons.append('contract_or_output_schema')
    tree=None;class_count=0;aliases=0;representatives=[]
    if not reasons:
        candidates=palette(spec);programs=[compile_trusted(source_allow_empty(c)) for c in candidates]
        sequences=[[]]+[[dict(kind='reset_scene')]+x for x in experiments(spec['family'])]
        probe_states=[state_for(spec['family'],spec['scene_seed'],s) for s in sequences]
        matrix=[[visible_signature(fn(copy.deepcopy(state))) for fn in programs] for state in probe_states]
        groups=defaultdict(list)
        for i in range(len(programs)):groups[tuple(row[i] for row in matrix)].append(i)
        representatives=[]
        for group in groups.values():
            first=group[0];representatives.append(first)
            if len(group)>1:
                base_values=[programs[first](copy.deepcopy(s)) for s in final]
                for other in group[1:]:
                    if any(not equivalent(programs[other](copy.deepcopy(s)),t) for s,t in zip(final,base_values)):
                        aliases+=1
        if aliases:reasons.append('rounded_or_probe_alias')
        else:
            collapsed=[[row[i] for i in representatives] for row in matrix]
            try:
                tree=decision_tree(collapsed,[len(s) for s in sequences]);class_count=len(representatives)
                if tree['cost']+3+2>spec['budget']:reasons.append('ideal_budget_two_spare')
            except ValueError:reasons.append('unidentifiable_palette')
    standardized=dict(spec,scene_seed=3410000)
    fingerprint=seal([tasks.canonical(solve(copy.deepcopy(s))) for s in grading_states(standardized,'final')])
    return {'accepted':not reasons,'reasons':sorted(set(reasons)),'spec_sha256':seal(spec),
        'source_sha256':seal(source),'neutral_parity':parity,'oracle_execution':runtime_ok,'independent_reference':independent,'reset_history_invariance':reset_ok,
        'protected_cases':len(final),'rule_checks':kills,'display_half_values':ties,'rounding_stable':rounding_stable,'contract_valid':contract_ok,
        'behavior_fingerprint':fingerprint,'ideal_catalogue_classes':class_count,
        'ideal_worst_experiment_actions':tree['cost'] if tree else None,
        'ideal_total_with_repair':tree['cost']+3 if tree else None,'spare_actions':spec['budget']-tree['cost']-3 if tree else None,
        'scope':'Exact footer-reading finite catalogue upper bound; arbitrary-program solvability is not asserted.',
        'private_tree':tree,'private_representatives':representatives}
